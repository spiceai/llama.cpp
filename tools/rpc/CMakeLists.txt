set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS AND UNIX AND NOT GGML_BACKEND_DL)
    add_executable(test-rpc-multi-server ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.cpp)
    target_link_libraries(test-rpc-multi-server PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-multi-server PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(
        NAME test-rpc-multi-server
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-multi-server>)
    set_property(TEST test-rpc-multi-server PROPERTY LABELS main)

    add_executable(test-rpc-comm-mesh ${PROJECT_SOURCE_DIR}/tests/test-rpc-comm-mesh.cpp)
    target_link_libraries(test-rpc-comm-mesh PRIVATE ggml ggml-rpc)
    add_test(
        NAME test-rpc-comm-mesh
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-comm-mesh.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-comm-mesh>)
    set_property(TEST test-rpc-comm-mesh PROPERTY LABELS main)

    add_executable(test-rpc-argmax-split ${PROJECT_SOURCE_DIR}/tests/test-rpc-argmax-split.cpp)
    target_link_libraries(test-rpc-argmax-split PRIVATE ggml ggml-rpc)
    add_test(
        NAME test-rpc-argmax-split
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-argmax-split.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-argmax-split>)
    set_property(TEST test-rpc-argmax-split PROPERTY LABELS main)

    add_executable(test-rpc-transport ${PROJECT_SOURCE_DIR}/tests/test-rpc-transport.cpp)
    target_link_libraries(test-rpc-transport PRIVATE ggml-rpc)
    target_include_directories(test-rpc-transport PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(NAME test-rpc-transport COMMAND $<TARGET_FILE:test-rpc-transport>)
    set_property(TEST test-rpc-transport PROPERTY LABELS main)
    add_executable(test-rpc-unreachable ${PROJECT_SOURCE_DIR}/tests/test-rpc-unreachable.cpp)
    target_link_libraries(test-rpc-unreachable PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-unreachable PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(NAME test-rpc-unreachable COMMAND test-rpc-unreachable)
    set_property(TEST test-rpc-unreachable PROPERTY LABELS main)
    # COPY_TENSOR into a zero-element destination outside its buffer is refused (CVE-2026-137567)
    add_executable(test-rpc-copy-tensor ${PROJECT_SOURCE_DIR}/tests/test-rpc-copy-tensor.cpp)
    target_link_libraries(test-rpc-copy-tensor PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-copy-tensor PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(NAME test-rpc-copy-tensor COMMAND test-rpc-copy-tensor)
    set_property(TEST test-rpc-copy-tensor PROPERTY LABELS main)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
