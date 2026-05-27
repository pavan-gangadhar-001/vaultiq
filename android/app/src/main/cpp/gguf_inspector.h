#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vaultiq {

std::string InspectGgufToJson(const std::string& path) noexcept;
std::string ValidateGgufLayerStreamingToJson(const std::string& path) noexcept;
std::string ReadGgufTokenEmbeddingToJson(const std::string& path, int token_id,
                                         int max_values) noexcept;
std::string RmsNormGgufTokenEmbeddingToJson(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    float epsilon, int max_values) noexcept;
std::string MatVecGgufRmsNormTokenEmbeddingToJson(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    const std::string& weight_tensor_name, float epsilon,
    int max_values) noexcept;
std::string RopeGgufMatVecRmsNormTokenEmbeddingToJson(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    const std::string& weight_tensor_name, float epsilon, int64_t position,
    int head_dim, double rope_theta, int max_values) noexcept;
std::string KvCacheGgufRmsNormTokenEmbeddingsToJson(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& norm_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t read_position, int head_dim,
    double rope_theta, int max_values) noexcept;
std::string AttentionGgufSingleHeadToJson(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, int head_index, double rope_theta,
    int max_values) noexcept;
std::string AttentionGgufMultiHeadToJson(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, double rope_theta,
    int max_values) noexcept;
std::string TransformerLayerGgufToJson(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& attn_norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name,
    const std::string& output_weight_tensor_name,
    const std::string& ffn_norm_tensor_name,
    const std::string& gate_weight_tensor_name,
    const std::string& up_weight_tensor_name,
    const std::string& down_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, double rope_theta,
    int max_values) noexcept;
std::string TransformerStackGgufToJson(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& attn_norm_tensor_suffix,
    const std::string& query_weight_tensor_suffix,
    const std::string& key_weight_tensor_suffix,
    const std::string& value_weight_tensor_suffix,
    const std::string& output_weight_tensor_suffix,
    const std::string& ffn_norm_tensor_suffix,
    const std::string& gate_weight_tensor_suffix,
    const std::string& up_weight_tensor_suffix,
    const std::string& down_weight_tensor_suffix, float epsilon,
    int64_t start_position, int read_position, int layer_count, int head_dim,
    double rope_theta, int max_values) noexcept;
std::string GenerateNextTokenGgufToJson(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& attn_norm_tensor_suffix,
    const std::string& query_weight_tensor_suffix,
    const std::string& key_weight_tensor_suffix,
    const std::string& value_weight_tensor_suffix,
    const std::string& output_weight_tensor_suffix,
    const std::string& ffn_norm_tensor_suffix,
    const std::string& gate_weight_tensor_suffix,
    const std::string& up_weight_tensor_suffix,
    const std::string& down_weight_tensor_suffix,
    const std::string& final_norm_tensor_name,
    const std::string& lm_head_tensor_name, float epsilon,
    int64_t start_position, int read_position, int layer_count, int head_dim,
    double rope_theta, int top_k, int max_values) noexcept;
std::string GenerateTokensGgufToJson(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& attn_norm_tensor_suffix,
    const std::string& query_weight_tensor_suffix,
    const std::string& key_weight_tensor_suffix,
    const std::string& value_weight_tensor_suffix,
    const std::string& output_weight_tensor_suffix,
    const std::string& ffn_norm_tensor_suffix,
    const std::string& gate_weight_tensor_suffix,
    const std::string& up_weight_tensor_suffix,
    const std::string& down_weight_tensor_suffix,
    const std::string& final_norm_tensor_name,
    const std::string& lm_head_tensor_name, float epsilon,
    int64_t start_position, int read_position, int layer_count, int head_dim,
    double rope_theta, int max_new_tokens, int top_k,
    int max_values) noexcept;
std::string CreateGgufSessionToJson(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& attn_norm_tensor_suffix,
    const std::string& query_weight_tensor_suffix,
    const std::string& key_weight_tensor_suffix,
    const std::string& value_weight_tensor_suffix,
    const std::string& output_weight_tensor_suffix,
    const std::string& ffn_norm_tensor_suffix,
    const std::string& gate_weight_tensor_suffix,
    const std::string& up_weight_tensor_suffix,
    const std::string& down_weight_tensor_suffix,
    const std::string& final_norm_tensor_name,
    const std::string& lm_head_tensor_name, float epsilon,
    int64_t start_position, int read_position, int layer_count, int head_dim,
    double rope_theta, int max_values) noexcept;
std::string DecodeGgufSessionToJson(uint64_t session_id, int top_k,
                                     int max_values) noexcept;
std::string CloseGgufSessionToJson(uint64_t session_id) noexcept;

}  // namespace vaultiq
