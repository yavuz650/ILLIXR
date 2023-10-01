LDFLAGS = $(shell pkg-config glew assimp --libs)
LDFLAGS := $(LDFLAGS) -L/usr/lib64/ -L/opt/ILLIXR/filament/lib/x86_64 -std=c++17 -Wl,--start-group -lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -lvkshaders -lfilamat -libl -lfilament-iblprefilter -lgltfio -lgltfio_core \
-lgeometry -ldracodec -lcamutils -limage -lktxreader -lshaders -lstb -lviewer -lmeshoptimizer -lbasis_transcoder -Wl,--end-group  \
-static-libgcc -static-libstdc++ -Wl,--gc-sections -fPIC -lc++ -lrt -ldl -lc++abi
CFLAGS = -fPIC 
include common/common.mk
