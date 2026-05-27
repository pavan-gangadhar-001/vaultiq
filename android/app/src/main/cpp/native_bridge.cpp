#include <jni.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gguf_inspector.h"

namespace {

jstring ErrorJson(JNIEnv* env, const char* message) {
  std::string json = "{\"ok\":false,\"error\":\"";
  json += message;
  json += "\"}";
  return env->NewStringUTF(json.c_str());
}

bool ReadJString(JNIEnv* env, jstring value, std::string& output) {
  if (value == nullptr) return false;
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) return false;
  output = chars;
  env->ReleaseStringUTFChars(value, chars);
  return true;
}

std::vector<int> ReadIntArray(JNIEnv* env, jintArray values) {
  if (values == nullptr) return {};
  const jsize value_count = env->GetArrayLength(values);
  std::vector<int> output(static_cast<size_t>(value_count));
  if (value_count > 0) {
    env->GetIntArrayRegion(values, 0, value_count,
                           reinterpret_cast<jint*>(output.data()));
  }
  return output;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_inspectGgufNative(
    JNIEnv* env, jobject /* thiz */, jstring path) {
  if (path == nullptr) {
    return env->NewStringUTF("{\"ok\":false,\"error\":\"Path is null\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }

  const std::string path_string(path_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  const std::string result = vaultiq::InspectGgufToJson(path_string);
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_kvCacheRmsNormTokenEmbeddingsNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jstring norm_tensor_name, jstring key_weight_tensor_name,
    jstring value_weight_tensor_name, jfloat epsilon, jlong start_position,
    jlong read_position, jint head_dim, jdouble rope_theta, jint max_values) {
  if (path == nullptr || token_ids == nullptr || norm_tensor_name == nullptr ||
      key_weight_tensor_name == nullptr || value_weight_tensor_name == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Path, token ids, norm tensor name, key "
        "tensor name, and value tensor name are required\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }
  const char* norm_chars = env->GetStringUTFChars(norm_tensor_name, nullptr);
  if (norm_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read norm tensor name\"}");
  }
  const char* key_chars =
      env->GetStringUTFChars(key_weight_tensor_name, nullptr);
  if (key_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read key tensor name\"}");
  }
  const char* value_chars =
      env->GetStringUTFChars(value_weight_tensor_name, nullptr);
  if (value_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    env->ReleaseStringUTFChars(key_weight_tensor_name, key_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read value tensor name\"}");
  }

  const jsize token_count = env->GetArrayLength(token_ids);
  std::vector<int> token_vector(static_cast<size_t>(token_count));
  if (token_count > 0) {
    env->GetIntArrayRegion(token_ids, 0, token_count,
                           reinterpret_cast<jint*>(token_vector.data()));
  }

  const std::string path_string(path_chars);
  const std::string norm_string(norm_chars);
  const std::string key_string(key_chars);
  const std::string value_string(value_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
  env->ReleaseStringUTFChars(key_weight_tensor_name, key_chars);
  env->ReleaseStringUTFChars(value_weight_tensor_name, value_chars);
  const std::string result = vaultiq::KvCacheGgufRmsNormTokenEmbeddingsToJson(
      path_string, token_vector, norm_string, key_string, value_string,
      static_cast<float>(epsilon), static_cast<int64_t>(start_position),
      static_cast<int64_t>(read_position), static_cast<int>(head_dim),
      static_cast<double>(rope_theta), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_attentionSingleHeadNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jint query_token_id, jstring norm_tensor_name,
    jstring query_weight_tensor_name, jstring key_weight_tensor_name,
    jstring value_weight_tensor_name, jfloat epsilon, jlong start_position,
    jlong query_position, jint read_position, jint attention_length,
    jint head_dim, jint head_index, jdouble rope_theta, jint max_values) {
  if (path == nullptr || token_ids == nullptr || norm_tensor_name == nullptr ||
      query_weight_tensor_name == nullptr || key_weight_tensor_name == nullptr ||
      value_weight_tensor_name == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Path, token ids, norm tensor name, query "
        "tensor name, key tensor name, and value tensor name are required\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }
  const char* norm_chars = env->GetStringUTFChars(norm_tensor_name, nullptr);
  if (norm_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read norm tensor name\"}");
  }
  const char* query_chars =
      env->GetStringUTFChars(query_weight_tensor_name, nullptr);
  if (query_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read query tensor name\"}");
  }
  const char* key_chars =
      env->GetStringUTFChars(key_weight_tensor_name, nullptr);
  if (key_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    env->ReleaseStringUTFChars(query_weight_tensor_name, query_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read key tensor name\"}");
  }
  const char* value_chars =
      env->GetStringUTFChars(value_weight_tensor_name, nullptr);
  if (value_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    env->ReleaseStringUTFChars(query_weight_tensor_name, query_chars);
    env->ReleaseStringUTFChars(key_weight_tensor_name, key_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read value tensor name\"}");
  }

  const jsize token_count = env->GetArrayLength(token_ids);
  std::vector<int> token_vector(static_cast<size_t>(token_count));
  if (token_count > 0) {
    env->GetIntArrayRegion(token_ids, 0, token_count,
                           reinterpret_cast<jint*>(token_vector.data()));
  }

  const std::string path_string(path_chars);
  const std::string norm_string(norm_chars);
  const std::string query_string(query_chars);
  const std::string key_string(key_chars);
  const std::string value_string(value_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
  env->ReleaseStringUTFChars(query_weight_tensor_name, query_chars);
  env->ReleaseStringUTFChars(key_weight_tensor_name, key_chars);
  env->ReleaseStringUTFChars(value_weight_tensor_name, value_chars);
  const std::string result = vaultiq::AttentionGgufSingleHeadToJson(
      path_string, token_vector, static_cast<int>(query_token_id), norm_string,
      query_string, key_string, value_string, static_cast<float>(epsilon),
      static_cast<int64_t>(start_position),
      static_cast<int64_t>(query_position), static_cast<int>(read_position),
      static_cast<int>(attention_length), static_cast<int>(head_dim),
      static_cast<int>(head_index), static_cast<double>(rope_theta),
      static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_attentionMultiHeadNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jint query_token_id, jstring norm_tensor_name,
    jstring query_weight_tensor_name, jstring key_weight_tensor_name,
    jstring value_weight_tensor_name, jfloat epsilon, jlong start_position,
    jlong query_position, jint read_position, jint attention_length,
    jint head_dim, jdouble rope_theta, jint max_values) {
  if (path == nullptr || token_ids == nullptr || norm_tensor_name == nullptr ||
      query_weight_tensor_name == nullptr || key_weight_tensor_name == nullptr ||
      value_weight_tensor_name == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Path, token ids, norm tensor name, query "
        "tensor name, key tensor name, and value tensor name are required\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }
  const char* norm_chars = env->GetStringUTFChars(norm_tensor_name, nullptr);
  if (norm_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read norm tensor name\"}");
  }
  const char* query_chars =
      env->GetStringUTFChars(query_weight_tensor_name, nullptr);
  if (query_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read query tensor name\"}");
  }
  const char* key_chars =
      env->GetStringUTFChars(key_weight_tensor_name, nullptr);
  if (key_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    env->ReleaseStringUTFChars(query_weight_tensor_name, query_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read key tensor name\"}");
  }
  const char* value_chars =
      env->GetStringUTFChars(value_weight_tensor_name, nullptr);
  if (value_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    env->ReleaseStringUTFChars(query_weight_tensor_name, query_chars);
    env->ReleaseStringUTFChars(key_weight_tensor_name, key_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read value tensor name\"}");
  }

  const jsize token_count = env->GetArrayLength(token_ids);
  std::vector<int> token_vector(static_cast<size_t>(token_count));
  if (token_count > 0) {
    env->GetIntArrayRegion(token_ids, 0, token_count,
                           reinterpret_cast<jint*>(token_vector.data()));
  }

  const std::string path_string(path_chars);
  const std::string norm_string(norm_chars);
  const std::string query_string(query_chars);
  const std::string key_string(key_chars);
  const std::string value_string(value_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
  env->ReleaseStringUTFChars(query_weight_tensor_name, query_chars);
  env->ReleaseStringUTFChars(key_weight_tensor_name, key_chars);
  env->ReleaseStringUTFChars(value_weight_tensor_name, value_chars);
  const std::string result = vaultiq::AttentionGgufMultiHeadToJson(
      path_string, token_vector, static_cast<int>(query_token_id), norm_string,
      query_string, key_string, value_string, static_cast<float>(epsilon),
      static_cast<int64_t>(start_position),
      static_cast<int64_t>(query_position), static_cast<int>(read_position),
      static_cast<int>(attention_length), static_cast<int>(head_dim),
      static_cast<double>(rope_theta), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_transformerLayerNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jint query_token_id, jstring attn_norm_tensor_name,
    jstring query_weight_tensor_name, jstring key_weight_tensor_name,
    jstring value_weight_tensor_name, jstring output_weight_tensor_name,
    jstring ffn_norm_tensor_name, jstring gate_weight_tensor_name,
    jstring up_weight_tensor_name, jstring down_weight_tensor_name,
    jfloat epsilon, jlong start_position, jlong query_position,
    jint read_position, jint attention_length, jint head_dim,
    jdouble rope_theta, jint max_values) {
  if (token_ids == nullptr) {
    return ErrorJson(env, "Path, token ids, and layer tensor names are required");
  }

  std::string path_string;
  std::string attn_norm_string;
  std::string query_string;
  std::string key_string;
  std::string value_string;
  std::string output_string;
  std::string ffn_norm_string;
  std::string gate_string;
  std::string up_string;
  std::string down_string;
  if (!ReadJString(env, path, path_string) ||
      !ReadJString(env, attn_norm_tensor_name, attn_norm_string) ||
      !ReadJString(env, query_weight_tensor_name, query_string) ||
      !ReadJString(env, key_weight_tensor_name, key_string) ||
      !ReadJString(env, value_weight_tensor_name, value_string) ||
      !ReadJString(env, output_weight_tensor_name, output_string) ||
      !ReadJString(env, ffn_norm_tensor_name, ffn_norm_string) ||
      !ReadJString(env, gate_weight_tensor_name, gate_string) ||
      !ReadJString(env, up_weight_tensor_name, up_string) ||
      !ReadJString(env, down_weight_tensor_name, down_string)) {
    return ErrorJson(env, "Unable to read path or layer tensor names");
  }

  const std::vector<int> token_vector = ReadIntArray(env, token_ids);
  const std::string result = vaultiq::TransformerLayerGgufToJson(
      path_string, token_vector, static_cast<int>(query_token_id),
      attn_norm_string, query_string, key_string, value_string, output_string,
      ffn_norm_string, gate_string, up_string, down_string,
      static_cast<float>(epsilon), static_cast<int64_t>(start_position),
      static_cast<int64_t>(query_position), static_cast<int>(read_position),
      static_cast<int>(attention_length), static_cast<int>(head_dim),
      static_cast<double>(rope_theta), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_transformerStackNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jstring attn_norm_tensor_suffix, jstring query_weight_tensor_suffix,
    jstring key_weight_tensor_suffix, jstring value_weight_tensor_suffix,
    jstring output_weight_tensor_suffix, jstring ffn_norm_tensor_suffix,
    jstring gate_weight_tensor_suffix, jstring up_weight_tensor_suffix,
    jstring down_weight_tensor_suffix, jfloat epsilon, jlong start_position,
    jint read_position, jint layer_count, jint head_dim, jdouble rope_theta,
    jint max_values) {
  if (token_ids == nullptr) {
    return ErrorJson(env, "Path, token ids, and stack tensor suffixes are required");
  }

  std::string path_string;
  std::string attn_norm_suffix;
  std::string query_suffix;
  std::string key_suffix;
  std::string value_suffix;
  std::string output_suffix;
  std::string ffn_norm_suffix;
  std::string gate_suffix;
  std::string up_suffix;
  std::string down_suffix;
  if (!ReadJString(env, path, path_string) ||
      !ReadJString(env, attn_norm_tensor_suffix, attn_norm_suffix) ||
      !ReadJString(env, query_weight_tensor_suffix, query_suffix) ||
      !ReadJString(env, key_weight_tensor_suffix, key_suffix) ||
      !ReadJString(env, value_weight_tensor_suffix, value_suffix) ||
      !ReadJString(env, output_weight_tensor_suffix, output_suffix) ||
      !ReadJString(env, ffn_norm_tensor_suffix, ffn_norm_suffix) ||
      !ReadJString(env, gate_weight_tensor_suffix, gate_suffix) ||
      !ReadJString(env, up_weight_tensor_suffix, up_suffix) ||
      !ReadJString(env, down_weight_tensor_suffix, down_suffix)) {
    return ErrorJson(env, "Unable to read path or stack tensor suffixes");
  }

  const std::vector<int> token_vector = ReadIntArray(env, token_ids);
  const std::string result = vaultiq::TransformerStackGgufToJson(
      path_string, token_vector, attn_norm_suffix, query_suffix, key_suffix,
      value_suffix, output_suffix, ffn_norm_suffix, gate_suffix, up_suffix,
      down_suffix, static_cast<float>(epsilon),
      static_cast<int64_t>(start_position), static_cast<int>(read_position),
      static_cast<int>(layer_count), static_cast<int>(head_dim),
      static_cast<double>(rope_theta), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_generateNextTokenNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jstring attn_norm_tensor_suffix, jstring query_weight_tensor_suffix,
    jstring key_weight_tensor_suffix, jstring value_weight_tensor_suffix,
    jstring output_weight_tensor_suffix, jstring ffn_norm_tensor_suffix,
    jstring gate_weight_tensor_suffix, jstring up_weight_tensor_suffix,
    jstring down_weight_tensor_suffix, jstring final_norm_tensor_name,
    jstring lm_head_tensor_name, jfloat epsilon, jlong start_position,
    jint read_position, jint layer_count, jint head_dim, jdouble rope_theta,
    jint top_k, jint max_values) {
  if (token_ids == nullptr) {
    return ErrorJson(env,
                     "Path, token ids, stack suffixes, final norm, and "
                     "lm_head tensor names are required");
  }

  std::string path_string;
  std::string attn_norm_suffix;
  std::string query_suffix;
  std::string key_suffix;
  std::string value_suffix;
  std::string output_suffix;
  std::string ffn_norm_suffix;
  std::string gate_suffix;
  std::string up_suffix;
  std::string down_suffix;
  std::string final_norm_string;
  std::string lm_head_string;
  if (!ReadJString(env, path, path_string) ||
      !ReadJString(env, attn_norm_tensor_suffix, attn_norm_suffix) ||
      !ReadJString(env, query_weight_tensor_suffix, query_suffix) ||
      !ReadJString(env, key_weight_tensor_suffix, key_suffix) ||
      !ReadJString(env, value_weight_tensor_suffix, value_suffix) ||
      !ReadJString(env, output_weight_tensor_suffix, output_suffix) ||
      !ReadJString(env, ffn_norm_tensor_suffix, ffn_norm_suffix) ||
      !ReadJString(env, gate_weight_tensor_suffix, gate_suffix) ||
      !ReadJString(env, up_weight_tensor_suffix, up_suffix) ||
      !ReadJString(env, down_weight_tensor_suffix, down_suffix) ||
      !ReadJString(env, final_norm_tensor_name, final_norm_string) ||
      !ReadJString(env, lm_head_tensor_name, lm_head_string)) {
    return ErrorJson(env,
                     "Unable to read path, stack suffixes, final norm, or "
                     "lm_head tensor names");
  }

  const std::vector<int> token_vector = ReadIntArray(env, token_ids);
  const std::string result = vaultiq::GenerateNextTokenGgufToJson(
      path_string, token_vector, attn_norm_suffix, query_suffix, key_suffix,
      value_suffix, output_suffix, ffn_norm_suffix, gate_suffix, up_suffix,
      down_suffix, final_norm_string, lm_head_string,
      static_cast<float>(epsilon), static_cast<int64_t>(start_position),
      static_cast<int>(read_position), static_cast<int>(layer_count),
      static_cast<int>(head_dim), static_cast<double>(rope_theta),
      static_cast<int>(top_k), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_generateTokensNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jstring attn_norm_tensor_suffix, jstring query_weight_tensor_suffix,
    jstring key_weight_tensor_suffix, jstring value_weight_tensor_suffix,
    jstring output_weight_tensor_suffix, jstring ffn_norm_tensor_suffix,
    jstring gate_weight_tensor_suffix, jstring up_weight_tensor_suffix,
    jstring down_weight_tensor_suffix, jstring final_norm_tensor_name,
    jstring lm_head_tensor_name, jfloat epsilon, jlong start_position,
    jint read_position, jint layer_count, jint head_dim, jdouble rope_theta,
    jint max_new_tokens, jint top_k, jint max_values) {
  if (token_ids == nullptr) {
    return ErrorJson(env,
                     "Path, token ids, stack suffixes, final norm, and "
                     "lm_head tensor names are required");
  }

  std::string path_string;
  std::string attn_norm_suffix;
  std::string query_suffix;
  std::string key_suffix;
  std::string value_suffix;
  std::string output_suffix;
  std::string ffn_norm_suffix;
  std::string gate_suffix;
  std::string up_suffix;
  std::string down_suffix;
  std::string final_norm_string;
  std::string lm_head_string;
  if (!ReadJString(env, path, path_string) ||
      !ReadJString(env, attn_norm_tensor_suffix, attn_norm_suffix) ||
      !ReadJString(env, query_weight_tensor_suffix, query_suffix) ||
      !ReadJString(env, key_weight_tensor_suffix, key_suffix) ||
      !ReadJString(env, value_weight_tensor_suffix, value_suffix) ||
      !ReadJString(env, output_weight_tensor_suffix, output_suffix) ||
      !ReadJString(env, ffn_norm_tensor_suffix, ffn_norm_suffix) ||
      !ReadJString(env, gate_weight_tensor_suffix, gate_suffix) ||
      !ReadJString(env, up_weight_tensor_suffix, up_suffix) ||
      !ReadJString(env, down_weight_tensor_suffix, down_suffix) ||
      !ReadJString(env, final_norm_tensor_name, final_norm_string) ||
      !ReadJString(env, lm_head_tensor_name, lm_head_string)) {
    return ErrorJson(env,
                     "Unable to read path, stack suffixes, final norm, or "
                     "lm_head tensor names");
  }

  const std::vector<int> token_vector = ReadIntArray(env, token_ids);
  const std::string result = vaultiq::GenerateTokensGgufToJson(
      path_string, token_vector, attn_norm_suffix, query_suffix, key_suffix,
      value_suffix, output_suffix, ffn_norm_suffix, gate_suffix, up_suffix,
      down_suffix, final_norm_string, lm_head_string,
      static_cast<float>(epsilon), static_cast<int64_t>(start_position),
      static_cast<int>(read_position), static_cast<int>(layer_count),
      static_cast<int>(head_dim), static_cast<double>(rope_theta),
      static_cast<int>(max_new_tokens), static_cast<int>(top_k),
      static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_createSessionNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jintArray token_ids,
    jstring attn_norm_tensor_suffix, jstring query_weight_tensor_suffix,
    jstring key_weight_tensor_suffix, jstring value_weight_tensor_suffix,
    jstring output_weight_tensor_suffix, jstring ffn_norm_tensor_suffix,
    jstring gate_weight_tensor_suffix, jstring up_weight_tensor_suffix,
    jstring down_weight_tensor_suffix, jstring final_norm_tensor_name,
    jstring lm_head_tensor_name, jfloat epsilon, jlong start_position,
    jint read_position, jint layer_count, jint head_dim, jdouble rope_theta,
    jint max_values) {
  if (token_ids == nullptr) {
    return ErrorJson(env,
                     "Path, token ids, stack suffixes, final norm, and "
                     "lm_head tensor names are required");
  }

  std::string path_string;
  std::string attn_norm_suffix;
  std::string query_suffix;
  std::string key_suffix;
  std::string value_suffix;
  std::string output_suffix;
  std::string ffn_norm_suffix;
  std::string gate_suffix;
  std::string up_suffix;
  std::string down_suffix;
  std::string final_norm_string;
  std::string lm_head_string;
  if (!ReadJString(env, path, path_string) ||
      !ReadJString(env, attn_norm_tensor_suffix, attn_norm_suffix) ||
      !ReadJString(env, query_weight_tensor_suffix, query_suffix) ||
      !ReadJString(env, key_weight_tensor_suffix, key_suffix) ||
      !ReadJString(env, value_weight_tensor_suffix, value_suffix) ||
      !ReadJString(env, output_weight_tensor_suffix, output_suffix) ||
      !ReadJString(env, ffn_norm_tensor_suffix, ffn_norm_suffix) ||
      !ReadJString(env, gate_weight_tensor_suffix, gate_suffix) ||
      !ReadJString(env, up_weight_tensor_suffix, up_suffix) ||
      !ReadJString(env, down_weight_tensor_suffix, down_suffix) ||
      !ReadJString(env, final_norm_tensor_name, final_norm_string) ||
      !ReadJString(env, lm_head_tensor_name, lm_head_string)) {
    return ErrorJson(env,
                     "Unable to read path, stack suffixes, final norm, or "
                     "lm_head tensor names");
  }

  const std::vector<int> token_vector = ReadIntArray(env, token_ids);
  const std::string result = vaultiq::CreateGgufSessionToJson(
      path_string, token_vector, attn_norm_suffix, query_suffix, key_suffix,
      value_suffix, output_suffix, ffn_norm_suffix, gate_suffix, up_suffix,
      down_suffix, final_norm_string, lm_head_string,
      static_cast<float>(epsilon), static_cast<int64_t>(start_position),
      static_cast<int>(read_position), static_cast<int>(layer_count),
      static_cast<int>(head_dim), static_cast<double>(rope_theta),
      static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_decodeSessionNative(
    JNIEnv* env, jobject /* thiz */, jlong session_id, jint top_k,
    jint max_values) {
  if (session_id <= 0) {
    return ErrorJson(env, "decodeGgufSession requires a positive sessionId");
  }

  const std::string result = vaultiq::DecodeGgufSessionToJson(
      static_cast<uint64_t>(session_id), static_cast<int>(top_k),
      static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_closeSessionNative(
    JNIEnv* env, jobject /* thiz */, jlong session_id) {
  if (session_id <= 0) {
    return ErrorJson(env, "closeGgufSession requires a positive sessionId");
  }

  const std::string result =
      vaultiq::CloseGgufSessionToJson(static_cast<uint64_t>(session_id));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_validateLayerStreamingNative(
    JNIEnv* env, jobject /* thiz */, jstring path) {
  if (path == nullptr) {
    return env->NewStringUTF("{\"ok\":false,\"error\":\"Path is null\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }

  const std::string path_string(path_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  const std::string result =
      vaultiq::ValidateGgufLayerStreamingToJson(path_string);
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_readTokenEmbeddingNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jint token_id,
    jint max_values) {
  if (path == nullptr) {
    return env->NewStringUTF("{\"ok\":false,\"error\":\"Path is null\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }

  const std::string path_string(path_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  const std::string result = vaultiq::ReadGgufTokenEmbeddingToJson(
      path_string, static_cast<int>(token_id), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_rmsNormTokenEmbeddingNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jint token_id,
    jstring norm_tensor_name, jfloat epsilon, jint max_values) {
  if (path == nullptr || norm_tensor_name == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Path and norm tensor name are required\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }
  const char* norm_chars = env->GetStringUTFChars(norm_tensor_name, nullptr);
  if (norm_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read norm tensor name\"}");
  }

  const std::string path_string(path_chars);
  const std::string norm_string(norm_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
  const std::string result = vaultiq::RmsNormGgufTokenEmbeddingToJson(
      path_string, static_cast<int>(token_id), norm_string,
      static_cast<float>(epsilon), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_matVecRmsNormTokenEmbeddingNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jint token_id,
    jstring norm_tensor_name, jstring weight_tensor_name, jfloat epsilon,
    jint max_values) {
  if (path == nullptr || norm_tensor_name == nullptr ||
      weight_tensor_name == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Path, norm tensor name, and weight tensor "
        "name are required\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }
  const char* norm_chars = env->GetStringUTFChars(norm_tensor_name, nullptr);
  if (norm_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read norm tensor name\"}");
  }
  const char* weight_chars = env->GetStringUTFChars(weight_tensor_name, nullptr);
  if (weight_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read weight tensor name\"}");
  }

  const std::string path_string(path_chars);
  const std::string norm_string(norm_chars);
  const std::string weight_string(weight_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
  env->ReleaseStringUTFChars(weight_tensor_name, weight_chars);
  const std::string result = vaultiq::MatVecGgufRmsNormTokenEmbeddingToJson(
      path_string, static_cast<int>(token_id), norm_string, weight_string,
      static_cast<float>(epsilon), static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pavganga_localdocqa_NativeGgufRunner_ropeMatVecRmsNormTokenEmbeddingNative(
    JNIEnv* env, jobject /* thiz */, jstring path, jint token_id,
    jstring norm_tensor_name, jstring weight_tensor_name, jfloat epsilon,
    jlong position, jint head_dim, jdouble rope_theta, jint max_values) {
  if (path == nullptr || norm_tensor_name == nullptr ||
      weight_tensor_name == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Path, norm tensor name, and weight tensor "
        "name are required\"}");
  }

  const char* path_chars = env->GetStringUTFChars(path, nullptr);
  if (path_chars == nullptr) {
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read path\"}");
  }
  const char* norm_chars = env->GetStringUTFChars(norm_tensor_name, nullptr);
  if (norm_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read norm tensor name\"}");
  }
  const char* weight_chars = env->GetStringUTFChars(weight_tensor_name, nullptr);
  if (weight_chars == nullptr) {
    env->ReleaseStringUTFChars(path, path_chars);
    env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
    return env->NewStringUTF(
        "{\"ok\":false,\"error\":\"Unable to read weight tensor name\"}");
  }

  const std::string path_string(path_chars);
  const std::string norm_string(norm_chars);
  const std::string weight_string(weight_chars);
  env->ReleaseStringUTFChars(path, path_chars);
  env->ReleaseStringUTFChars(norm_tensor_name, norm_chars);
  env->ReleaseStringUTFChars(weight_tensor_name, weight_chars);
  const std::string result =
      vaultiq::RopeGgufMatVecRmsNormTokenEmbeddingToJson(
          path_string, static_cast<int>(token_id), norm_string, weight_string,
          static_cast<float>(epsilon), static_cast<int64_t>(position),
          static_cast<int>(head_dim), static_cast<double>(rope_theta),
          static_cast<int>(max_values));
  return env->NewStringUTF(result.c_str());
}
