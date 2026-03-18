# go-rnnoise

<a href="https://pkg.go.dev/github.com/MarcosTypeAP/go-rnnoise"><img src="https://pkg.go.dev/badge/github.com/MarcosTypeAP/go-rnnoise.svg" alt="Go Reference"></a>

A Go wrapper around [RNNoise](https://github.com/xiph/rnnoise), a recurrent neural network based noise suppression library.

See [RNNoise license](RNNOISE_LICENSE).

## Installation

```sh
go get github.com/MarcosTypeAP/go-rnnoise
```

## Audio format

RNNoise processes audio at **48 kHz** in fixed-size frames of **480 s16-bit samples** (`FrameSize`).

| Function | Expected sample range |
|---|---|
| `ProcessFrame` | `float32` in `[-32768.0, 32767.0]` |
| `ProcessFrameNormalized` | `float32` in `[-1.0, 1.0]` |

## Usage

### Quick start (global state)

The simplest way to suppress noise is through the package-level global state, which is initialized automatically on the first call:

```go
vad := rnnoise.ProcessFrame(out, in)
fmt.Printf("Voice activity probability: %.2f\n", vad)
```

> **Note:** The global state is **not thread-safe**. For concurrent processing, allocate an explicit `DenoiseState` per goroutine.

### Explicit state

```go
state, err := rnnoise.New(nil) // nil = default built-in model
if err != nil {
    log.Fatal(err)
}

vad := state.ProcessFrame(out, in)
```

### Normalized floats

```go
// Samples in [-1.0, 1.0]
vad := rnnoise.ProcessFrameNormalized(out, in)
```

> **Warning:** `ProcessFrameNormalized` temporarily mutates `in` by scaling it up and back down. Use `ProcessFrame` with pre-scaled samples if `in` must remain unmodified.

### In-place processing

`in` and `out` may point to the same slice:

```go
vad := rnnoise.ProcessFrame(buf, buf)
```

## Custom models

A custom RNNoise model can be loaded from a file path, an open `*os.File`, or a byte slice.

```go
// From a file path
model, err := rnnoise.LoadModelFromFilename("weights_blob.bin")
if err != nil {
    log.Fatal(err)
}
defer rnnoise.ModelFree(&model)

state, err := rnnoise.New(&model)
```

```go
// From a byte slice (e.g. embedded with go:embed)
model := rnnoise.LoadModelFromBuffer(data)
defer rnnoise.ModelFree(&model)
```

A model may be shared across multiple `DenoiseState` instances. It must not be freed until all states that reference it have been discarded.

## Voice activity detection (VAD)

`ProcessFrame` and `ProcessFrameNormalized` both return a VAD probability in `[0, 1]`. Values close to `1` indicate that speech is likely present in the frame.

```go
vad := rnnoise.ProcessFrame(out, in)
if vad > 0.5 {
    fmt.Println("speech detected")
}
```

## Thread safety

Each `DenoiseState` is independent and safe to use from a single goroutine. Do **not** share a `DenoiseState` between goroutines. The global state (nil state argument) is **not thread-safe**.
