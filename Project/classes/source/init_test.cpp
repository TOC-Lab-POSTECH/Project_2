#include "init_test.h"
#include <iostream>

init_test::INIT_INSTANCE::INIT_INSTANCE(){}

void init_test::INIT_INSTANCE::print(){
    std::cout << this->hello;
}
