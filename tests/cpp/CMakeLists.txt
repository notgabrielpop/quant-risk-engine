add_executable(qre_tests test_gbm.cpp)
target_link_libraries(qre_tests PRIVATE qre_engine Catch2::Catch2WithMain)
catch_discover_tests(qre_tests)
