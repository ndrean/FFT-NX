# Discrete Fourrier Transform with Nx

```elixir
Mix.install([
  {:nx, "~> 0.9.1"},
  {:kino, "~> 0.14.2"},
  {:kino_vega_lite, "~> 0.1.11"},
  {:exla, "~> 0.9.1"}
], config: [nx: [default_backend: EXLA.Backend]]
)

Nx.Defn.global_default_options(compiler: EXLA, client: :host)
```

## Complex numerical helper

```elixir
defmodule Ncx do
  import Nx.Defn

  defn i(), do: Nx.Constants.i()

  defn new(x,y), do: x + i() * y

  # square norm
  defn sq_norm(z), do: Nx.conjugate(z) |> Nx.dot(z) |> Nx.real()

  defn minus(x,y), do: Nx.add(x, Nx.dot(y, -1))
end
```

## Using complex numbers

Take the complex number $1+i$.

Since `Nx` brings in the library `Complex`, we define a complex number by `Complex.new/1`:

```elixir
one = Complex.new(1,1)
```

Its norm is $ \sqrt{2}$, and angle is $\pi/4$.

Its polar form is:

$ \sqrt{2}\big(\cos(\pi/4) + i\sin(\pi/4)\big) = \sqrt{2} \exp^{i \pi/4}$.

We can use `Complex.from_polar/2`:

```elixir
sqrt_2 = :math.sqrt(2)
a = :math.pi()/4

cx_one = Complex.from_polar(sqrt_2, a)
```

```elixir
{
  Complex.real(cx_one),
  Complex.imag(cx_one)
}
```

We can also use `Nx` to build the Euler-De Moivre formula:

```elixir
t_one = Nx.complex(Nx.tensor(1), Nx.tensor(1))
```

## Example of DFT

Suppose we have a signal and we can sample it at 50Hz, meaning accepting 50 signals per second.

We wnat ot discover this signal. We will run a Fast Fourrier Taansform

<https://en.wikipedia.org/wiki/Discrete_Fourier_transform>

<!-- livebook:{"break_markdown":true} -->

Lets build a signal that we want to discover!

It is composed of the sum of two sinusoid signals, one at 5Hz and the second at 15Hz with amplitudes (1, 0.5).

$\sin(2\pi*5*t / f_s) + \frac12 \sin(2\pi*15*t/f_s)$

We can do something like this "by hand":

```elixir
fs = 64
sampling = Nx.linspace(0, fs, n: fs, endpoint: false, type: {:u, 64})
pi = Nx.Constants.pi()

_sampling =
  Nx.add(
    Nx.sin(Nx.dot(2, pi)|> Nx.dot(5) |> Nx.dot(sampling) |> Nx.divide(fs)),
    Nx.sin(Nx.dot(2, pi)|> Nx.dot(15) |> Nx.dot(sampling) |> Nx.divide(fs)) |> Nx.dot(0.5)
  )
```

> It is easier to perform calculations in a defined numerical function with `defn` like below, so we adopt it. Morevoer, the code will be compiled by the backend, thus more efficient.

We build a linear space of 50 equally spaced points with `Nx.linspace`. For each value of the discrete time $t$, we will calculate the "hidden" signal at $t$ in our "Signal" module. This will give us the sampling.

```elixir
defmodule Signal do
  import Nx.Defn

  defn pi(), do: Nx.Constants.pi()


  defn sample(opts) do
    fs = opts[:fs]
    sampling = Nx.linspace(0, fs, n: fs, endpoint: false, type: {:u, 64})
    f1 = 5
    f2 = 15
    Nx.sin(2 * pi() * f1 * sampling / fs) + 1/2 * Nx.sin(2 * pi() * f2 * sampling / fs)
  end

  # we want to apply Ncx.sq_norm to each element of the tensor so we vectorize it
  # we do not need the norm (square root but only the square of the norm
  defn amplitudes(t) do
    Nx.vectorize(t, :x) |> Ncx.sq_norm()
  end
end
```

We obtain 64 samples per second in the list below

> Note they are all real values

```elixir
fs = 2**6
samples = Signal.sample(fs: fs)
```

We transform theses equally spaced values by `Nx.fft` and obtain a list of Fourrier coeficients: they are complex values indexed by the frequency.

```elixir
dft = Nx.fft(samples)
```

This gives us the following information:

> The Fourrier coefficient as the position $i$ of this list gives the information of the amplitude (the norm of this coefficient) of the sampled signal at the frequency $i$.

Our frequency resolution - the accuracy - is limited by the Nyquist frequency, which is half the sampling rate, thus up to 32Hz.

We will therefor limit the display to 32Hz, which are the 32 first points returned by the FFT.

```elixir
f_max = div(fs, 2)

amplitudes =  Signal.amplitudes(dft)

data1 = %{
  x: (for i<- 0..f_max, do: i),
  y: Nx.to_list(amplitudes) |> Enum.take(f_max)
}
```

```elixir
VegaLite.new(width: 600, height: 300)
|> VegaLite.data_from_values(data1, only: ["x", "y"])
|> VegaLite.mark(:point)
|> VegaLite.encode_field(:x, "x", type: :quantitative)
|> VegaLite.encode_field(:y, "y", type: :quantitative)
```

![image](https://github.com/user-attachments/assets/6dc625c1-718c-462f-90fb-976eee0d5a16)

We see a peak at 5Hz and one at 15Hz with amplitude about 4 times less. Since we took the square of the norm (to speed up computations), this means the amplitutde is half. This is indeed our incomming signa!! 🎉
