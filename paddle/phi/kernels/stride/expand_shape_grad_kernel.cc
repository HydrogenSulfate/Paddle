// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reduce_as_kernel.h"
#include "paddle/phi/kernels/view_grad_kernel.h"
#include "paddle/phi/kernels/view_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename T, typename Context>
void ExpandStridedGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& out_grad,
                             const IntArray& dims /*unused*/,
                             DenseTensor* x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  dev_ctx.template Alloc<T>(x_grad);
  ReduceAsKernel<T, Context>(dev_ctx, out_grad, x, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(expand_shape_grad,
                   CPU,
                   STRIDED,
                   phi::ExpandStridedGradKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(expand_shape_grad,
                   GPU,
                   STRIDED,
                   phi::ExpandStridedGradKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
