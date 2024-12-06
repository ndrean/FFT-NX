async function init(ctx, html) {
  ctx.root.innerHTML = html;

  const height = 256,
    width = 256,
    video = document.getElementById("invid"),
    renderCanvas = document.getElementById("canvas"),
    renderCanvasCtx = canvas.getContext("2d"),
    offscreen = new OffscreenCanvas(width, height),
    offscreenCtx = offscreen.getContext("2d");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width, height },
    });
    video.srcObject = stream;
    video.addEventListener("loadedmetadata", () => {
      video.play();
      video.requestVideoFrameCallback(processFrame);
    });

    const processFrame = async (now, metadata) => {
      offscreenCtx.drawImage(video, 0, 0, width, height);
      const imageData = offscreenCtx.getImageData(0, 0, width, height);
      const rawData = imageData.data;
      console.log(rawData.buffer.byteLength);
      ctx.pushEvent("new frame", [{}, rawData.buffer]);
      video.requestVideoFrameCallback(processFrame);
    };
  } catch (err) {
    console.error("Webcam access error:", err);
  }

  function rawDataToPng(binary, width, height) {
    const imageData = renderCanvasCtx.createImageData(width, height);
    imageData.data.set(new Uint8ClampedArray(binary));
    renderCanvasCtx.putImageData(imageData, 0, 0);

    return new Promise((resolve) => {
      renderCanvas.toBlob((blob) => resolve(blob), "image/png");
    });
  }

  ctx.handleEvent("processed_frame", async ([{}, binary]) => {
    await createImageBitmap(new Blob([binary], { type: "image/png" })).then(
      (bitmap) => renderCanvasCtx.drawImage(bitmap, 0, 0)
    );
  });
}

export async function init(ctx, html) {
  ctx.root.innerHTML = html;

  const width = 256,
    height = 256,
    video = document.getElementById("gpuinvid"),
    gpuCanvas = document.getElementById("gpucanvas"),
    gpuCanvasCtx = gpuCanvas.getContext("webgpu"),
    offscreen = new OffscreenCanvas(width, height),
    offscreenCtx = offscreen.getContext("2d", { willReadFrequently: true });

  // Create a WebGPU device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Create a WebGPU canvas context
  gpuCanvasCtx.configure({
    device: device,
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Create buffers and pipelines
  const videoTexture = device.createTexture({
    size: [width, height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const gaussianShaderCode = `
    @group(0) @binding(0) var inputTexture: texture_2d<f32>;
    @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

    const kernelSize: i32 = 5;
    const kernel: array<f32, 5> = array<f32, 5>(
        0.06136, 0.24477, 0.38774, 0.24477, 0.06136
    );

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let coords = vec2<i32>(global_id.xy);
        let texSize = textureDimensions(inputTexture);

        if (coords.x >= i32(texSize.x) || coords.y >= i32(texSize.y)) {
            return;
        }

        var color: vec4<f32> = vec4<f32>(0.0);

        // Horizontal blur
        for (var i: i32 = -kernelSize / 2; i <= kernelSize / 2; i = i + 1) {
            let offsetCoords = coords + vec2<i32>(i, 0);
            let clampedCoords = clamp(
                offsetCoords,
                vec2<i32>(0),
                vec2<i32>(i32(texSize.x - 1), i32(texSize.y - 1))
            );
            color += kernel[i + kernelSize / 2] * textureLoad(inputTexture, clampedCoords, 0);
        }

        // Write the result to the output texture
        textureStore(outputTexture, coords, color);
    }
    `;

  // Compile and use the shader
  const shaderModule = device.createShaderModule({ code: gaussianShaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: "write-only", format: "rgba8unorm" },
      },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [] }),
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: inputTexture.createView() },
      { binding: 1, resource: outputTexture.createView() },
    ],
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width, height },
  });
  video.srcObject = stream;
  // let frameCounter = 0;

  video.play();

  // Process video frames with WebGPU
  const processFrame = async () => {
    offscreenCtx.drawImage(video, 0, 0, width, height);
    const imageData = offscreenCtx.getImageData(0, 0, width, height);

    // Upload image data to WebGPU texture
    device.queue.writeTexture(
      { texture: videoTexture },
      imageData.data,
      { bytesPerRow: width * 4 },
      { width, height, depthOrArrayLayers: 1 }
    );

    // Dispatch compute shader for Gaussian blur
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    // Copy the output texture to the canvas
    const renderPassDescriptor = {
      colorAttachments: [
        {
          view: gpuCanvasCtx.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    };

    const renderEncoder = device.createCommandEncoder();
    const renderPass = renderEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.end();
    device.queue.submit([renderEncoder.finish()]);
  };

  processFrame();
}
