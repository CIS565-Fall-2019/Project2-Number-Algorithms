set(SOURCE_FILES
    "common.h"
    "common.cu"
    "mlp.h"
    "mlp.cu"
    )

cuda_add_library(character_recognition
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )

target_link_libraries(character_recognition ${CUDA_LIBRARIES})
target_link_libraries(character_recognition ${CUDA_CUBLAS_LIBRARIES})
