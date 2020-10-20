package parallel

// #cgo CFLAGS: -I ${SRCDIR} -I ${SRCDIR}/cgotorch/libtorch/include
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch -Wl,-rpath ${SRCDIR}/cgotorch -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/cgotorch/libtorch/lib -Wl,-rpath ${SRCDIR}/cgotorch/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include "cgotorch/cgotorch.h"
import (
	"C"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
	"github.com/wangkuiyi/gotorch/nn"
)

// export goModuleForward
func goModuleForward(m unsafe.Pointer, input C.Tensor) C.Tensor {
	module := Module()
	forward := reflect.ValueOf(s.Modules[0]).MethodByName("Forward")
	return forward.Call(getReflectInputs(torch.Tensor{&inputs})).T
}

func DataParallel(m nn.IModule, input torch.Tensor, devices []torch.Device, outputDevice torch.Device, dim int64) {
}
