cmake_minimum_required(VERSION 3.8)
project(rm_armor_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(USE_CUDA_DETCTOR "Use CUDA for detection" OFF)
option(USE_OPENVINO_DETCTOR "Use OpenVINO for detection" OFF)

if(USE_CUDA_DETCTOR)
    add_definitions(-DUSE_CUDA_DETCTOR)
    # 设置默认的 ONNX Runtime 根目录，如果用户没有手动提供
    if(NOT ONNXRUNTIME_ROOTDIR)
    if(WIN32)
        # Windows 系统下设置默认的 ONNX Runtime 路径
        set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
    else()
        # 非 Windows 系统下设置默认路径
        set(ONNXRUNTIME_ROOTDIR "${CMAKE_SOURCE_DIR}/onnxruntime")
    endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_ROOTDIR}/lib64/cmake/onnxruntime")
    # 添加包含路径
    include_directories(
    ${ONNXRUNTIME_ROOTDIR}/include
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
    ${OPENVINO_ROOTDIR}/runtime/include
    )
    find_package(onnxruntime REQUIRED)
endif()

if(USE_OPENVINO_DETCTOR)
    add_definitions(-DUSE_OPENVINO_DETCTOR)
    set(OpenVINO_LIBRARIES "/opt/openvino/runtime/lib/intel64/libopenvino.so")
    set(OpenVINO_INCLUDE_DIRS "/opt/openvino/runtime/include")
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
#find_package(onnxruntime REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

ament_auto_find_build_dependencies()

if(USE_CUDA_DETCTOR)
  ament_auto_add_library(${PROJECT_NAME} SHARED
    DIRECTORY src/detector_node src/cuda_detector.cpp
  )
  target_include_directories(${PROJECT_NAME} PUBLIC include/rm_armor_detector/detector_node include/rm_armor_detector)
elseif(USE_OPENVINO_DETCTOR)
  ament_auto_add_library(${PROJECT_NAME} SHARED
    DIRECTORY src/detector_node src/openvino_detector.cpp
  )
  target_include_directories(${PROJECT_NAME} PUBLIC include/rm_armor_detector/detector_node include/rm_armor_detector ${OpenVINO_INCLUDE_DIRS})
else()
  ament_auto_add_library(${PROJECT_NAME} SHARED
    DIRECTORY src/detector_node
  )
  target_include_directories(${PROJECT_NAME} PUBLIC include/rm_armor_detector/detector_node)
endif()

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_armor_detector::ArmorDetectorNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  model
)
