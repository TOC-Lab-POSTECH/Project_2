#ifndef init_test_h
#define init_test_h

#include <string>

namespace init_test {
class INIT_INSTANCE
{
private:
    std::string hello = "HELLO\n";
public:
    INIT_INSTANCE();
    void print();
};

}

#endif /* init_test_h */
