#pragma once

class LogicGate {
public:
    enum class Type {
        ZERO,      // 0
        AND,       // 1
        A_NOT_B,   // 2
        A,         // 3
        B_NOT_A,   // 4
        B,         // 5
        XOR,       // 6
        OR,        // 7
        NOR,       // 8
        XNOR,      // 9
        NOT_B,     // 10
        A_OR_NOT_B,// 11
        NOT_A,     // 12
        NOT_A_OR_B,// 13
        NAND,      // 14
        ONE        // 15
    };

    static bool apply(Type type, bool a, bool b) {
        switch (type) {
            case Type::ZERO: return false;
            case Type::AND: return a && b;
            case Type::A_NOT_B: return a && !b;
            case Type::A: return a;
            case Type::B_NOT_A: return !a && b;
            case Type::B: return b;
            case Type::XOR: return a != b;
            case Type::OR: return a || b;
            case Type::NOR: return !(a || b);
            case Type::XNOR: return a == b;
            case Type::NOT_B: return !b;
            case Type::A_OR_NOT_B: return a || !b;
            case Type::NOT_A: return !a;
            case Type::NOT_A_OR_B: return !a || b;
            case Type::NAND: return !(a && b);
            case Type::ONE: return true;
            default: return false;
        }
    }
};
