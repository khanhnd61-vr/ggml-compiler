#pragma once

#include <vector>

#include "ggml.h"
#include "program.h"

namespace ggmlc {

// Builds the ggml tensor for one IR node. Checks the preconditions ggml would otherwise
// abort on, and checks the result's type and shape against the node spec; both failures
// throw std::runtime_error naming the node. `gpu` lets ops pick a backend-specific kernel.
ggml_tensor * emit_node(ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & src, bool gpu);

}  // namespace ggmlc
