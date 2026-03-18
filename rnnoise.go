// Package rnnoise provides a Go wrapper around the RNNoise C library, a
// recurrent neural network based noise suppression library.
//
// For more information, look at the [RNNoise repository].
//
// RNNoise processes audio at 48 kHz in fixed-size frames of [FrameSize]
// samples. Samples are expected as 16-bit signed integers scaled to the range
// [-32768.0, 32767.0], or as normalized 32-bit floats in [-1, 1]
// when using [ProcessFrameNormalized].
//
// # Basic usage
//
// The simplest way to use the package is through the global state, which is
// initialized automatically on the first call to [ProcessFrame] or
// [ProcessFrameNormalized]:
//
//	vad, err := rnnoise.ProcessFrame(nil, out, in)
//
// For more control, allocate an explicit [DenoiseState]:
//
//	state, err := rnnoise.New(nil) // nil = default built-in model
//	if err != nil { ... }
//	vad := state.ProcessFrame(out, in)
//
// # Custom models
//
// A custom RNNoise model can be loaded from a file or buffer:
//
//	model, err := rnnoise.LoadModelFromFilename("weights_blob.bin")
//	if err != nil { ... }
//	defer rnnoise.ModelFree(&model)
//
//	state, err := rnnoise.New(&model)
//
// # Thread safety
//
// Each [DenoiseState] is independent and may be used concurrently from
// separate goroutines as long as a single state is not shared between them.
// The global state accessed via a nil state argument is NOT thread-safe.
//
// [RNNoise repository]: https://github.com/xiph/rnnoise
package rnnoise

// #cgo linux CFLAGS: -I${SRCDIR}/rnnoise/linux/include
// #cgo linux LDFLAGS: -L${SRCDIR}/rnnoise/linux/lib -Wl,-Bstatic -lrnnoise -Wl,-Bdynamic -lm
//
// #cgo windows CFLAGS: -I${SRCDIR}/rnnoise/windows/include
// #cgo windows LDFLAGS: -L${SRCDIR}/rnnoise/windows/lib -Wl,-Bstatic -lrnnoise -Wl,-Bdynamic -lm
//
// #include <stdlib.h>
// #include "rnnoise.h"
import "C"
import (
	"fmt"
	"io"
	"os"
	"unsafe"
)

const FrameSize = 480

const _FreqSize = (FrameSize + 1)
const _NBBands = 32

const (
	_PitchMaxPeriod = 768
	_PitchFrameSize = 960
	_PitchBufSize   = _PitchMaxPeriod + _PitchFrameSize
)

const (
	_Conv1StateSize = 65 * 2
	_Conv2StateSize = 128 * 2
)

const (
	_GRU1StateSize = 384
	_GRU2StateSize = 384
	_GRU3StateSize = 384
)

type _LinearLayer struct {
	bias         *float32
	subias       *float32
	weights      *int8
	floatWeights *float32
	weightsIdx   *int32
	diag         *float32
	scale        *float32
	nbInputs     int32
	nbOutputs    int32
}

type _RNNoise struct {
	conv1         _LinearLayer
	conv2         _LinearLayer
	gru1Input     _LinearLayer
	gru1Recurrent _LinearLayer
	gru2Input     _LinearLayer
	gru2Recurrent _LinearLayer
	gru3Input     _LinearLayer
	gru3Recurrent _LinearLayer
	denseOut      _LinearLayer
	vadDense      _LinearLayer
}

type _RNNState struct {
	conv1State [_Conv1StateSize]float32
	conv2State [_Conv2StateSize]float32
	gru1State  [_GRU1StateSize]float32
	gru2State  [_GRU2StateSize]float32
	gru3State  [_GRU3StateSize]float32
}

type _kissFftScala = float32

type _kissFftCpx struct {
	r _kissFftScala
	i _kissFftScala
}

// DenoiseState is bound to a single audio stream; do not reuse the same
// state for unrelated streams. A zero-value DenoiseState must be initialized
// with [Init] before use; prefer [New] or [Make] for allocation.
type DenoiseState struct {
	model        _RNNoise
	arch         int32
	analysisMem  [FrameSize]float32
	memid        int32
	synthesisMem [FrameSize]float32
	pitchBuf     [_PitchBufSize]float32
	pitchEnhBuf  [_PitchBufSize]float32
	lastGain     float32
	lastPeriod   int32
	memHpX       [2]float32
	lastg        [_NBBands]float32
	rnn          _RNNState
	delayedX     [_FreqSize]_kissFftCpx
	delayedP     [_FreqSize]_kissFftCpx
	delayedEx    [_NBBands]float32
	delayedEp    [_NBBands]float32
	delayedExp   [_NBBands]float32
}

// RNNModel holds a loaded RNNoise model. The model data is stored in a
// C-allocated buffer (blob) and must be freed with [ModelFree] when no
// longer needed.
//
// A model may be shared across multiple [DenoiseState] instances. It must not
// be freed until all states that reference it have been discarded.
type RNNModel struct {
	constBlob uintptr
	blob      unsafe.Pointer
	blobLen   int32
	file      uintptr
}

func init() {
	sizeC := int(C.rnnoise_get_size())
	sizeGo := int(unsafe.Sizeof(DenoiseState{}))
	if sizeGo != sizeC {
		panic(fmt.Errorf("DenoiseState struct size mismatch: C=%d GO=%d", sizeC, sizeGo))
	}

	frameSizeC := int(C.rnnoise_get_frame_size())
	if FrameSize != frameSizeC {
		panic(fmt.Errorf("FrameSize mismatch: C=%d GO=%d", sizeC, FrameSize))
	}
}

var globalDenoiseState *DenoiseState

var ErrLoaded = fmt.Errorf("global denoise state is already loaded")
var ErrUnloaded = fmt.Errorf("global denoise state is not loaded")

// Load initializes the global [DenoiseState] with the given model.
// If model is nil the default built-in model is used.
//
// Load is called automatically by [ProcessFrame] and [ProcessFrameNormalized]
// when their state argument is nil, so explicit calls are only needed when
// pre-loading is desirable (e.g. to surface errors early).
//
// Returns [ErrLoaded] if the global state is already initialized. Call
// [Unload] first to replace it.
func Load(model *RNNModel) error {
	if globalDenoiseState != nil {
		return ErrLoaded
	}
	globalDenoiseState = new(DenoiseState)
	status := C.rnnoise_init((*C.DenoiseState)(unsafe.Pointer(globalDenoiseState)), (*C.RNNModel)(unsafe.Pointer(model)))
	if status != 0 {
		return fmt.Errorf("could not load rnnoise")
	}
	return nil
}

// Unload releases the global [DenoiseState], allowing it to be re-initialized
// by a subsequent call to [Load] or the first call to [ProcessFrame].
//
// Returns [ErrUnloaded] if the global state is not currently initialized.
func Unload() error {
	if globalDenoiseState == nil {
		return ErrUnloaded
	}
	globalDenoiseState = nil
	return nil
}

// Init initializes a pre-allocated [DenoiseState].
// If model is nil the default model is used.
func Init(state *DenoiseState, model *RNNModel) error {
	status := C.rnnoise_init((*C.DenoiseState)(unsafe.Pointer(state)), (*C.RNNModel)(unsafe.Pointer(model)))
	if status != 0 {
		return fmt.Errorf("could not initialize state")
	}
	return nil
}

// New allocates a new [DenoiseState] on the heap and initializes it with the
// given model.
// If model is nil the default built-in model is used.
func New(model *RNNModel) (*DenoiseState, error) {
	state := new(DenoiseState)
	if err := Init(state, model); err != nil {
		return nil, err
	}
	return state, nil
}

// Make returns an initialized [DenoiseState] by value.
// If model is nil the default built-in model is used.
func Make(model *RNNModel) (DenoiseState, error) {
	state := DenoiseState{}
	if err := Init(&state, model); err != nil {
		return DenoiseState{}, err
	}
	return state, nil
}

// ProcessFrame denoise a frame of samples.
//
// This uses the global [DenoiseState]; it is initialized automatically on the
// first call (not thread-safe). For concurrent use, allocate an explicit state.
//
// Samples must be 16-bit signed integers scaled to [-32768.0, 32767.0] at
// 48 kHz. For normalized floats in [-1, 1] use [ProcessFrameNormalized].
// Both in and out must have length >= [FrameSize]; the function panics
// otherwise. in and out may point to the same slice for in-place processing.
//
// Returns the voice activity detection (VAD) probability in [0, 1], where
// values close to 1 indicate speech is likely present.
func ProcessFrame(out, in []float32) (vad float32) {
	return processFrame(nil, out, in)
}

// ProcessFrame is like the package-level [ProcessFrame] but uses s as the
// [DenoiseState].
func (s *DenoiseState) ProcessFrame(out, in []float32) (vad float32) {
	return processFrame(s, out, in)
}

func processFrame(state *DenoiseState, out, in []float32) (vad float32) {
	if state == nil {
		if globalDenoiseState == nil {
			if err := Load(nil); err != nil {
				panic(fmt.Errorf("loading global state: %v", err))
			}
		}
		state = globalDenoiseState
	}
	if len(in) < FrameSize {
		panic(fmt.Errorf("len(in) must be >= %d, got %d", FrameSize, len(in)))
	}
	if len(out) < FrameSize {
		panic(fmt.Errorf("len(out) must be >= %d, got %d", FrameSize, len(out)))
	}

	vad = float32(C.rnnoise_process_frame((*C.DenoiseState)(unsafe.Pointer(state)), (*C.float)(unsafe.Pointer(&out[0])), (*C.float)(unsafe.Pointer(&in[0]))))
	return vad
}

// ProcessFrameNormalized is like [ProcessFrame], but samples must be normalized
// 32-bit floats in [-1, 1].
//
// The first [FrameSize] elements of in are temporarily scaled up to the range
// expected by RNNoise and then scaled back afterwards:
//
//	in[i] *= 32767  // before processing
//	in[i] /= 32767  // after processing
//
// As a result, in is mutated during the call. Use [ProcessFrame] with
// pre-scaled samples if you need in to remain unmodified.
func ProcessFrameNormalized(out, in []float32) (vad float32) {
	return processFrameNormalized(nil, out, in)
}

// ProcessFrameNormalized is like the package-level [ProcessFrameNormalized]
// but uses s as the [DenoiseState].
func (s *DenoiseState) ProcessFrameNormalized(out, in []float32) (vad float32) {
	return processFrameNormalized(s, out, in)
}

func processFrameNormalized(state *DenoiseState, out, in []float32) (vad float32) {
	if state == nil {
		if globalDenoiseState == nil {
			if err := Load(nil); err != nil {
				panic(fmt.Errorf("loading global state: %v", err))
			}
		}
		state = globalDenoiseState
	}
	if len(in) < FrameSize {
		panic(fmt.Errorf("len(in) must be >= %d, got %d", FrameSize, len(in)))
	}
	if len(out) < FrameSize {
		panic(fmt.Errorf("len(out) must be >= %d, got %d", FrameSize, len(out)))
	}

	for i := range FrameSize {
		in[i] *= 32767
	}

	vad = float32(C.rnnoise_process_frame((*C.DenoiseState)(unsafe.Pointer(state)), (*C.float)(unsafe.Pointer(&out[0])), (*C.float)(unsafe.Pointer(&in[0]))))

	for i := range FrameSize {
		in[i] /= 32767
	}

	return vad
}

// LoadModelFromBuffer loads a model from a byte slice by copying its contents
// into C-managed memory. The returned model must be freed with [ModelFree]
// when it is no longer needed.
func LoadModelFromBuffer(buf []byte) RNNModel {
	return RNNModel{
		blobLen: int32(len(buf)),
		blob:    C.CBytes(buf),
	}
}

// LoadModelFromFile loads a model from an open [*os.File] by reading its
// entire contents into C-managed memory. The file may be closed immediately
// after this call returns. The returned model must be freed with [ModelFree].
func LoadModelFromFile(file *os.File) (RNNModel, error) {
	info, err := file.Stat()
	if err != nil {
		return RNNModel{}, fmt.Errorf("getting info about the file: %s: %w", file.Name(), err)
	}
	bufLen := info.Size()

	cBuf := C.malloc((C.size_t)(bufLen))

	n, err := file.Read(unsafe.Slice((*byte)(cBuf), bufLen))
	if n != int(bufLen) {
		panic(fmt.Errorf("readed data length mismatch: expected %d, got %d", bufLen, n))
	}
	if err != nil && err != io.EOF {
		return RNNModel{}, fmt.Errorf("reading file: %s: %w", file.Name(), err)
	}

	model := RNNModel{
		blob:    cBuf,
		blobLen: int32(bufLen),
	}
	return model, nil
}

// LoadModelFromFilename opens the file at the given path, loads its contents
// as a model, and closes the file. It is a convenience wrapper around
// [LoadModelFromFile]. The returned model must be freed with [ModelFree].
func LoadModelFromFilename(filename string) (RNNModel, error) {
	file, err := os.Open(filename)
	if err != nil {
		return RNNModel{}, fmt.Errorf("opening file: %s: %w", filename, err)
	}
	defer file.Close()
	model, err := LoadModelFromFile(file)
	if err != nil {
		return RNNModel{}, fmt.Errorf("processing file content: %s: %w", file.Name(), err)
	}
	return model, nil
}

// ModelFree releases the C-managed memory held by the model. It must be
// called after all [DenoiseState] instances that reference this model have
// been discarded, as those states read from the model's blob at processing
// time.
//
// Panics if the model has already been freed.
func ModelFree(model *RNNModel) {
	if model.blob == nil {
		panic("model already freed")
	}
	C.free(model.blob)
	model.blob = nil
}
