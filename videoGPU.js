export async function init(ctx, html) {
  ctx.root.innerHTML = html;

  const width = 256;
  const height = 256;

  const adapter = await navigator.gpu.requestAdapter();
  console.log(adapter.info);
  const device = await adapter.requestDevice();
  const gpuCanvas = document.getElementById("gpucanvas");
  const gpuCanvasCtx = gpuCanvas.getContext("webgpu");
  const offscreen = new OffscreenCanvas(width, height);
  const offscreenCtx = offscreen.getContext("2d", { willReadFrequently: true });

  gpuCanvasCtx.configure({
    device,
    format: "bgra8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const inputTexture = device.createTexture({
    size: [width, height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const outputTexture = device.createTexture({
    size: [width, height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const fragmentShaderCode = `
    @group(0) @binding(0) var outputTexture: texture_2d<f32>;
    @group(0) @binding(2) var outputSampler: sampler;
  
    @fragment
    fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let size = vec2f(256.0, 256.0);
      let uv = pos.xy / size;
      return textureSample(outputTexture, outputSampler, uv);
    }
  `;

  const computeShaderCode = `
    @group(0) @binding(0) var inputTexture: texture_2d<f32>;
    @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(2) var ourSampler: sampler;  // Added sampler

    /*
    const kernelSize: i32 = 5;
    const kernel: array<f32, 5> = array<f32, 5>(
      0.1, 0.15, 0.5, 0.15, 0.1
    );
    */

    const kernelSize: i32 = 9;
    const kernel: array<f32, 9> = array<f32, 9>(
      0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229
    );


  
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let coords = vec2<i32>(global_id.xy);
      let texSize = textureDimensions(inputTexture);
      
      if (coords.x >= i32(texSize.x) || coords.y >= i32(texSize.y)) {
        return;
      }
  
      var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      
      // 2D Gaussian blur
      for (var y: i32 = -kernelSize / 2; y <= kernelSize / 2; y = y + 1) {
        for (var x: i32 = -kernelSize / 2; x <= kernelSize / 2; x = x + 1) {
          let offsetCoords = coords + vec2<i32>(x, y);
          let clampedCoords = clamp(
            offsetCoords,
            vec2<i32>(0),
            vec2<i32>(i32(texSize.x - 1), i32(texSize.y - 1))
          );
          let weight = kernel[x + kernelSize / 2] * kernel[y + kernelSize / 2];
          color += weight * textureLoad(inputTexture, clampedCoords, 0);
        }
      }

      textureStore(outputTexture, coords, color);
    }
  `;

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: "write-only", format: "rgba8unorm" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
    ],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: inputTexture.createView() },
      { binding: 1, resource: outputTexture.createView() },
      { binding: 2, resource: sampler }, // Add sampler
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({ code: computeShaderCode }),
      entryPoint: "main",
    },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    vertex: {
      module: device.createShaderModule({
        code: `
        @vertex
        fn main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
          var pos = array<vec2f, 6>(
            vec2f(-1.0, -1.0),
            vec2f(1.0, -1.0),
            vec2f(-1.0, 1.0),
            vec2f(-1.0, 1.0),
            vec2f(1.0, -1.0),
            vec2f(1.0, 1.0)
          );
          return vec4f(pos[vertexIndex], 0.0, 1.0);
        }
      `,
      }),
      entryPoint: "main",
    },
    fragment: {
      module: device.createShaderModule({ code: fragmentShaderCode }),
      entryPoint: "main",
      targets: [{ format: "bgra8unorm" }],
    },
    primitive: { topology: "triangle-list" },
  });

  const video = document.getElementById("gpuinvid");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width, height },
  });
  video.srcObject = stream;
  video.play();

  function processFrame() {
    offscreenCtx.drawImage(video, 0, 0, width, height);
    const imageData = offscreenCtx.getImageData(0, 0, width, height);

    device.queue.writeTexture(
      { texture: inputTexture },
      imageData.data,
      { bytesPerRow: width * 4 },
      { width, height }
    );

    const commandEncoder = device.createCommandEncoder();

    // Compute pass
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    computePass.end();

    // Render pass
    const renderPassDescriptor = {
      colorAttachments: [
        {
          view: gpuCanvasCtx.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    };

    const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(processFrame);
  }

  processFrame();
}
