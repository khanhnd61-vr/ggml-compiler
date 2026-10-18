#pragma once

#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "program.h"

namespace ggml_compiler {

// Builds the ggml tensor for one IR node. Checks the preconditions ggml would otherwise
// abort on, and checks the result's type and shape against the node spec; both failures
// throw std::runtime_error naming the node. `backend` lets ops pick a backend-specific kernel, or
// another expansion where the backend cannot run the usual one.
ggml_tensor * emit_node(ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & src,
                        ggml_backend_t backend);

}  // namespace ggml_compiler
