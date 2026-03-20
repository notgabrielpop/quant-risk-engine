add_executable(test_engine test_gbm.cpp test_heston.cpp test_rough_heston.cpp test_variance_reduction.cpp test_risk.cpp test_main.cpp)
target_link_libraries(test_engine PRIVATE quant_engine)
