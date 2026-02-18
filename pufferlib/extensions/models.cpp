// models.cpp - Torch-free utilities (PufTensor + pure math)
// Included by pufferlib.cpp inside namespace pufferlib

// Fast clip_grad_norm_ for contiguous grad buffer using PufTensor kernel
void clip_grad_norm_(PufTensor& grad, float max_norm, float* scratch, cudaStream_t stream) {
    if (grad.data == nullptr || grad.numel == 0) return;
    puf_clip_grad_norm(grad, max_norm, scratch, stream);
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}
