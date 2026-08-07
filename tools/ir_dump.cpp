// ir_dump — print a model's Sip IR (the platform's stable representation).
//
// Reads only the header/metadata/tensor-directory (no weight data, no
// inference), builds the SipModel via the importer, and prints it. This is the
// inspect/trace primitive: what does the runtime think this model IS?
//
//   ir_dump <model.gguf> [--summary]
#include "llm/runtime.h"   // open_model
#include "llm/sip_ir.h"

#include <cstdio>
#include <string>

using namespace llm;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--summary]\n", argv[0]);
        return 2;
    }
    bool summary_only = false;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--summary") summary_only = true;

    try {
        auto src = open_model(argv[1]);
        SipModel m = import_model(*src);
        if (summary_only) printf("%s\n", m.summary().c_str());
        else              printf("%s", m.to_json().c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
