#include "gguf_inspector.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace vaultiq {
namespace {

constexpr uint32_t kGgufMagic = 0x46554747;  // "GGUF", little endian.
constexpr uint64_t kMaxReasonableStringBytes = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMaxTensorDims = 16;

enum class MetadataType : uint32_t {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

struct TensorLayout {
  uint64_t block_size = 0;
  uint64_t type_size = 0;
};

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  uint32_t type = 0;
  uint64_t relative_offset = 0;
  uint64_t absolute_offset = 0;
  uint64_t element_count = 0;
  uint64_t byte_size = 0;
  bool has_known_byte_size = false;
};

struct GgufIndex {
  std::string path;
  uint64_t file_size = 0;
  uint32_t version = 0;
  uint64_t tensor_count = 0;
  uint64_t metadata_count = 0;
  uint64_t alignment = 32;
  uint64_t data_start = 0;
  uint64_t parameter_count = 0;
  std::string architecture;
  std::string arch_prefix;
  std::map<std::string, std::string> metadata;
  std::map<std::string, uint64_t> tensor_types;
  std::vector<TensorInfo> tensors;
};

struct LayerSummary {
  int index = 0;
  uint64_t tensor_count = 0;
  uint64_t span_start = 0;
  uint64_t span_bytes = 0;
  uint64_t largest_tensor_bytes = 0;
};

class Reader {
 public:
  explicit Reader(const std::string& path) : file_(path, std::ios::binary) {
    if (!file_) {
      throw std::runtime_error("Unable to open GGUF file");
    }
    file_.seekg(0, std::ios::end);
    file_size_ = static_cast<uint64_t>(file_.tellg());
    file_.seekg(0, std::ios::beg);
  }

  uint64_t file_size() const { return file_size_; }

  uint64_t position() {
    const auto pos = file_.tellg();
    if (pos < 0) {
      throw std::runtime_error("Failed to read file position");
    }
    return static_cast<uint64_t>(pos);
  }

  template <typename T>
  T read_pod() {
    T value{};
    file_.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file_) {
      throw std::runtime_error("Unexpected end of GGUF file");
    }
    return value;
  }

  std::string read_string() {
    const uint64_t length = read_pod<uint64_t>();
    if (length > kMaxReasonableStringBytes) {
      throw std::runtime_error("GGUF string is unreasonably large");
    }
    std::string value;
    value.resize(static_cast<size_t>(length));
    if (length == 0) return value;
    file_.read(value.data(), static_cast<std::streamsize>(length));
    if (!file_) {
      throw std::runtime_error("Unexpected end while reading GGUF string");
    }
    return value;
  }

  void skip(uint64_t byte_count) {
    if (byte_count == 0) return;
    const uint64_t pos = position();
    if (byte_count > file_size_ || pos > file_size_ - byte_count) {
      throw std::runtime_error("GGUF skip moves beyond file end");
    }
    file_.seekg(static_cast<std::streamoff>(byte_count), std::ios::cur);
    if (!file_) {
      throw std::runtime_error("Failed to skip GGUF bytes");
    }
  }

 private:
  std::ifstream file_;
  uint64_t file_size_ = 0;
};

class FileDescriptor {
 public:
  explicit FileDescriptor(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      throw std::runtime_error("Unable to open GGUF for mmap: " +
                               std::string(std::strerror(errno)));
    }
  }

  ~FileDescriptor() {
    if (fd_ >= 0) close(fd_);
  }

  int get() const { return fd_; }

 private:
  int fd_ = -1;
};

class MappedRegion {
 public:
  MappedRegion(int fd, uint64_t offset, uint64_t length) {
    if (length == 0) {
      throw std::runtime_error("Cannot mmap an empty layer span");
    }
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (page_size <= 0) {
      throw std::runtime_error("Unable to determine mmap page size");
    }
    const uint64_t page = static_cast<uint64_t>(page_size);
    const uint64_t aligned_offset = offset & ~(page - 1);
    const uint64_t delta = offset - aligned_offset;
    const uint64_t map_len_64 = delta + length;
    if (map_len_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      throw std::runtime_error("Layer span is too large to map");
    }

    map_len_ = static_cast<size_t>(map_len_64);
    base_ = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd,
                 static_cast<off_t>(aligned_offset));
    if (base_ == MAP_FAILED) {
      base_ = nullptr;
      throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    }
    data_ = static_cast<uint8_t*>(base_) + delta;
    madvise(base_, map_len_, MADV_SEQUENTIAL);

    // Touch one byte so the validation proves the mapping is readable without
    // forcing the whole model layer into memory.
    volatile uint8_t first_byte = data_[0];
    (void)first_byte;
  }

  MappedRegion(const MappedRegion&) = delete;
  MappedRegion& operator=(const MappedRegion&) = delete;

  ~MappedRegion() {
    if (base_ != nullptr) {
      madvise(base_, map_len_, MADV_DONTNEED);
      munmap(base_, map_len_);
    }
  }

 private:
  void* base_ = nullptr;
  uint8_t* data_ = nullptr;
  size_t map_len_ = 0;
};

std::string JsonEscape(const std::string& input) {
  std::ostringstream out;
  for (const unsigned char c : input) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

std::string Quote(const std::string& input) {
  return "\"" + JsonEscape(input) + "\"";
}

std::string MetadataTypeName(uint32_t type) {
  switch (static_cast<MetadataType>(type)) {
    case MetadataType::kUint8:
      return "uint8";
    case MetadataType::kInt8:
      return "int8";
    case MetadataType::kUint16:
      return "uint16";
    case MetadataType::kInt16:
      return "int16";
    case MetadataType::kUint32:
      return "uint32";
    case MetadataType::kInt32:
      return "int32";
    case MetadataType::kFloat32:
      return "float32";
    case MetadataType::kBool:
      return "bool";
    case MetadataType::kString:
      return "string";
    case MetadataType::kArray:
      return "array";
    case MetadataType::kUint64:
      return "uint64";
    case MetadataType::kInt64:
      return "int64";
    case MetadataType::kFloat64:
      return "float64";
  }
  return "unknown";
}

std::string TensorTypeName(uint32_t type) {
  switch (type) {
    case 0:
      return "F32";
    case 1:
      return "F16";
    case 2:
      return "Q4_0";
    case 3:
      return "Q4_1";
    case 6:
      return "Q5_0";
    case 7:
      return "Q5_1";
    case 8:
      return "Q8_0";
    case 9:
      return "Q8_1";
    case 10:
      return "Q2_K";
    case 11:
      return "Q3_K";
    case 12:
      return "Q4_K";
    case 13:
      return "Q5_K";
    case 14:
      return "Q6_K";
    case 15:
      return "Q8_K";
    case 16:
      return "IQ2_XXS";
    case 17:
      return "IQ2_XS";
    case 18:
      return "IQ3_XXS";
    case 19:
      return "IQ1_S";
    case 20:
      return "IQ4_NL";
    case 21:
      return "IQ3_S";
    case 22:
      return "IQ2_S";
    case 23:
      return "IQ4_XS";
    case 24:
      return "I8";
    case 25:
      return "I16";
    case 26:
      return "I32";
    case 27:
      return "I64";
    case 28:
      return "F64";
    case 29:
      return "IQ1_M";
    case 30:
      return "BF16";
    default:
      return "UNKNOWN_" + std::to_string(type);
  }
}

bool TensorLayoutForType(uint32_t type, TensorLayout& layout) {
  switch (type) {
    case 0:   // F32
      layout = {1, 4};
      return true;
    case 1:   // F16
      layout = {1, 2};
      return true;
    case 2:   // Q4_0
      layout = {32, 18};
      return true;
    case 3:   // Q4_1
      layout = {32, 20};
      return true;
    case 6:   // Q5_0
      layout = {32, 22};
      return true;
    case 7:   // Q5_1
      layout = {32, 24};
      return true;
    case 8:   // Q8_0
      layout = {32, 34};
      return true;
    case 9:   // Q8_1
      layout = {32, 40};
      return true;
    case 10:  // Q2_K
      layout = {256, 84};
      return true;
    case 11:  // Q3_K
      layout = {256, 110};
      return true;
    case 12:  // Q4_K
      layout = {256, 144};
      return true;
    case 13:  // Q5_K
      layout = {256, 176};
      return true;
    case 14:  // Q6_K
      layout = {256, 210};
      return true;
    case 15:  // Q8_K
      layout = {256, 292};
      return true;
    case 24:  // I8
      layout = {1, 1};
      return true;
    case 25:  // I16
      layout = {1, 2};
      return true;
    case 26:  // I32
      layout = {1, 4};
      return true;
    case 27:  // I64
      layout = {1, 8};
      return true;
    case 28:  // F64
      layout = {1, 8};
      return true;
    case 30:  // BF16
      layout = {1, 2};
      return true;
    default:
      return false;
  }
}

uint64_t AlignOffset(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    throw std::runtime_error("Invalid GGUF alignment");
  }
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

uint64_t CheckedAdd(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::runtime_error("GGUF size overflow");
  }
  return left + right;
}

uint64_t CheckedMultiply(uint64_t left, uint64_t right) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::runtime_error("GGUF size overflow");
  }
  return left * right;
}

size_t CheckedSize(uint64_t value, const std::string& message) {
  if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(message);
  }
  return static_cast<size_t>(value);
}

uint64_t CeilDivide(uint64_t value, uint64_t divisor) {
  if (divisor == 0) {
    throw std::runtime_error("Invalid zero divisor");
  }
  return value / divisor + (value % divisor == 0 ? 0 : 1);
}

bool TensorDataBytes(uint32_t type, uint64_t elements, uint64_t& byte_size) {
  TensorLayout layout;
  if (!TensorLayoutForType(type, layout)) return false;
  byte_size = CheckedMultiply(CeilDivide(elements, layout.block_size),
                              layout.type_size);
  return true;
}

bool TensorRowBytes(uint32_t type, uint64_t row_elements, uint64_t& byte_size) {
  return TensorDataBytes(type, row_elements, byte_size);
}

uint16_t ReadLeU16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
}

float ReadLeF32(const uint8_t* data) {
  uint32_t bits = static_cast<uint32_t>(data[0]) |
                  (static_cast<uint32_t>(data[1]) << 8) |
                  (static_cast<uint32_t>(data[2]) << 16) |
                  (static_cast<uint32_t>(data[3]) << 24);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float Float16ToFloat(uint16_t half) {
  const uint32_t sign = (static_cast<uint32_t>(half & 0x8000)) << 16;
  uint32_t exponent = (half >> 10) & 0x1f;
  uint32_t mantissa = half & 0x03ff;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float BFloat16ToFloat(uint16_t bfloat) {
  const uint32_t bits = static_cast<uint32_t>(bfloat) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<uint8_t> ReadFileBytes(int fd, uint64_t offset, uint64_t length) {
  if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("Requested GGUF read is too large");
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(length));
  size_t copied = 0;
  while (copied < bytes.size()) {
    const ssize_t read_count =
        pread(fd, bytes.data() + copied, bytes.size() - copied,
              static_cast<off_t>(offset + copied));
    if (read_count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("pread failed: " +
                               std::string(std::strerror(errno)));
    }
    if (read_count == 0) {
      throw std::runtime_error("Unexpected EOF while reading GGUF tensor");
    }
    copied += static_cast<size_t>(read_count);
  }
  return bytes;
}

uint64_t ScalarTypeBytes(uint32_t type) {
  switch (static_cast<MetadataType>(type)) {
    case MetadataType::kUint8:
    case MetadataType::kInt8:
    case MetadataType::kBool:
      return 1;
    case MetadataType::kUint16:
    case MetadataType::kInt16:
      return 2;
    case MetadataType::kUint32:
    case MetadataType::kInt32:
    case MetadataType::kFloat32:
      return 4;
    case MetadataType::kUint64:
    case MetadataType::kInt64:
    case MetadataType::kFloat64:
      return 8;
    case MetadataType::kString:
    case MetadataType::kArray:
      return 0;
  }
  throw std::runtime_error("Unknown GGUF metadata scalar type");
}

std::string ReadScalarAsString(Reader& reader, uint32_t type) {
  switch (static_cast<MetadataType>(type)) {
    case MetadataType::kUint8:
      return std::to_string(reader.read_pod<uint8_t>());
    case MetadataType::kInt8:
      return std::to_string(reader.read_pod<int8_t>());
    case MetadataType::kUint16:
      return std::to_string(reader.read_pod<uint16_t>());
    case MetadataType::kInt16:
      return std::to_string(reader.read_pod<int16_t>());
    case MetadataType::kUint32:
      return std::to_string(reader.read_pod<uint32_t>());
    case MetadataType::kInt32:
      return std::to_string(reader.read_pod<int32_t>());
    case MetadataType::kFloat32: {
      std::ostringstream out;
      out << reader.read_pod<float>();
      return out.str();
    }
    case MetadataType::kBool:
      return reader.read_pod<uint8_t>() == 0 ? "false" : "true";
    case MetadataType::kString:
      return reader.read_string();
    case MetadataType::kUint64:
      return std::to_string(reader.read_pod<uint64_t>());
    case MetadataType::kInt64:
      return std::to_string(reader.read_pod<int64_t>());
    case MetadataType::kFloat64: {
      std::ostringstream out;
      out << reader.read_pod<double>();
      return out.str();
    }
    case MetadataType::kArray:
      throw std::runtime_error("Nested scalar read requested for array");
  }
  throw std::runtime_error("Unknown GGUF metadata type");
}

void SkipScalar(Reader& reader, uint32_t type) {
  if (static_cast<MetadataType>(type) == MetadataType::kString) {
    const uint64_t length = reader.read_pod<uint64_t>();
    reader.skip(length);
    return;
  }
  const uint64_t bytes = ScalarTypeBytes(type);
  if (bytes == 0) {
    throw std::runtime_error("Unsupported nested GGUF array metadata");
  }
  reader.skip(bytes);
}

std::string ReadMetadataValue(Reader& reader, uint32_t type) {
  if (static_cast<MetadataType>(type) != MetadataType::kArray) {
    return ReadScalarAsString(reader, type);
  }

  const uint32_t element_type = reader.read_pod<uint32_t>();
  const uint64_t length = reader.read_pod<uint64_t>();
  if (static_cast<MetadataType>(element_type) == MetadataType::kString) {
    for (uint64_t i = 0; i < length; ++i) {
      SkipScalar(reader, element_type);
    }
  } else {
    const uint64_t bytes = CheckedMultiply(length, ScalarTypeBytes(element_type));
    reader.skip(bytes);
  }
  return "array<" + MetadataTypeName(element_type) + ">[" +
         std::to_string(length) + "]";
}

uint64_t ParseUnsigned(const std::map<std::string, std::string>& metadata,
                       const std::string& key, uint64_t fallback) {
  const auto it = metadata.find(key);
  if (it == metadata.end()) return fallback;
  try {
    return static_cast<uint64_t>(std::stoull(it->second));
  } catch (...) {
    return fallback;
  }
}

double ParseDouble(const std::map<std::string, std::string>& metadata,
                   const std::string& key, double fallback) {
  const auto it = metadata.find(key);
  if (it == metadata.end()) return fallback;
  try {
    const double value = std::stod(it->second);
    return std::isfinite(value) ? value : fallback;
  } catch (...) {
    return fallback;
  }
}

std::string FindMetadataString(const std::map<std::string, std::string>& metadata,
                               const std::string& key) {
  const auto it = metadata.find(key);
  return it == metadata.end() ? "" : it->second;
}

void AppendJsonField(std::ostringstream& out, const std::string& name,
                     const std::string& value, bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":" << Quote(value);
}

void AppendJsonBoolField(std::ostringstream& out, const std::string& name,
                         bool value, bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":" << (value ? "true" : "false");
}

void AppendJsonNumberField(std::ostringstream& out, const std::string& name,
                           uint64_t value, bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":" << value;
}

void AppendJsonSignedNumberField(std::ostringstream& out,
                                 const std::string& name, int64_t value,
                                 bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":" << value;
}

void AppendJsonFloatField(std::ostringstream& out, const std::string& name,
                          double value, bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":";
  if (std::isfinite(value)) {
    out << std::setprecision(9) << value;
  } else {
    out << "null";
  }
}

void AppendJsonIntArrayField(std::ostringstream& out, const std::string& name,
                             const std::vector<int>& values, bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ",";
    out << values[i];
  }
  out << "]";
}

void AppendJsonFloatArrayField(std::ostringstream& out,
                               const std::string& name,
                               const std::vector<float>& values, size_t count,
                               bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":[";
  const size_t resolved_count = std::min(count, values.size());
  for (size_t i = 0; i < resolved_count; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(values[i])) {
      out << std::setprecision(9) << values[i];
    } else {
      out << "null";
    }
  }
  out << "]";
}

void AppendJsonDoubleArrayField(std::ostringstream& out,
                                const std::string& name,
                                const std::vector<double>& values,
                                size_t count, bool& first) {
  if (!first) out << ",";
  first = false;
  out << Quote(name) << ":[";
  const size_t resolved_count = std::min(count, values.size());
  for (size_t i = 0; i < resolved_count; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(values[i])) {
      out << std::setprecision(9) << values[i];
    } else {
      out << "null";
    }
  }
  out << "]";
}

int LayerIndexFromTensorName(const std::string& name) {
  constexpr const char* kBlockPrefix = "blk.";
  constexpr size_t kBlockPrefixLength = 4;
  if (name.rfind(kBlockPrefix, 0) != 0) return -1;

  size_t cursor = kBlockPrefixLength;
  if (cursor >= name.size() || name[cursor] < '0' || name[cursor] > '9') {
    return -1;
  }

  int layer = 0;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    const int digit = name[cursor] - '0';
    if (layer > (std::numeric_limits<int>::max() - digit) / 10) return -1;
    layer = layer * 10 + digit;
    ++cursor;
  }
  if (cursor >= name.size() || name[cursor] != '.') return -1;
  return layer;
}

const TensorInfo* FindTensor(const GgufIndex& index, const std::string& name) {
  for (const TensorInfo& tensor : index.tensors) {
    if (tensor.name == name) return &tensor;
  }
  return nullptr;
}

std::vector<float> DecodeTensorRow(int fd, const TensorInfo& tensor,
                                   uint64_t row_index,
                                   uint64_t row_elements) {
  uint64_t row_bytes = 0;
  if (!TensorRowBytes(tensor.type, row_elements, row_bytes)) {
    throw std::runtime_error("Unsupported token embedding tensor type: " +
                             TensorTypeName(tensor.type));
  }
  const uint64_t row_offset =
      CheckedAdd(tensor.absolute_offset, CheckedMultiply(row_index, row_bytes));
  const uint64_t row_end = CheckedAdd(row_offset, row_bytes);
  const uint64_t tensor_end =
      CheckedAdd(tensor.absolute_offset, tensor.byte_size);
  if (row_end > tensor_end) {
    throw std::runtime_error("Token embedding row extends beyond tensor data");
  }

  const std::vector<uint8_t> bytes = ReadFileBytes(fd, row_offset, row_bytes);
  std::vector<float> values;
  values.reserve(static_cast<size_t>(row_elements));

  switch (tensor.type) {
    case 0: {  // F32
      for (uint64_t i = 0; i < row_elements; ++i) {
        values.push_back(ReadLeF32(bytes.data() + i * 4));
      }
      return values;
    }
    case 1: {  // F16
      for (uint64_t i = 0; i < row_elements; ++i) {
        values.push_back(Float16ToFloat(ReadLeU16(bytes.data() + i * 2)));
      }
      return values;
    }
    case 8: {  // Q8_0: half scale followed by 32 signed quantized values.
      constexpr uint64_t kBlockSize = 32;
      constexpr uint64_t kTypeSize = 34;
      const uint64_t block_count = CeilDivide(row_elements, kBlockSize);
      for (uint64_t block = 0; block < block_count; ++block) {
        const uint8_t* block_data = bytes.data() + block * kTypeSize;
        const float scale = Float16ToFloat(ReadLeU16(block_data));
        const uint64_t block_base = block * kBlockSize;
        const uint64_t block_values =
            std::min<uint64_t>(kBlockSize, row_elements - block_base);
        for (uint64_t i = 0; i < block_values; ++i) {
          const auto quantized =
              static_cast<int8_t>(*(block_data + 2 + i));
          values.push_back(scale * static_cast<float>(quantized));
        }
      }
      return values;
    }
    case 30: {  // BF16
      for (uint64_t i = 0; i < row_elements; ++i) {
        values.push_back(BFloat16ToFloat(ReadLeU16(bytes.data() + i * 2)));
      }
      return values;
    }
    default:
      throw std::runtime_error("Unsupported token embedding tensor type: " +
                               TensorTypeName(tensor.type));
  }
}

std::vector<float> DecodeTensorVector(int fd, const TensorInfo& tensor) {
  if (tensor.element_count == 0) {
    throw std::runtime_error("Cannot decode an empty tensor vector");
  }
  return DecodeTensorRow(fd, tensor, 0, tensor.element_count);
}

struct RmsNormOutput {
  std::vector<float> values;
  double mean_square = 0.0;
  double inv_rms = 0.0;
};

RmsNormOutput ApplyRmsNorm(const std::vector<float>& input,
                           const std::vector<float>& weights,
                           float epsilon) {
  if (input.empty()) {
    throw std::runtime_error("Cannot RMSNorm an empty vector");
  }
  if (input.size() != weights.size()) {
    throw std::runtime_error("RMSNorm input and weight sizes do not match");
  }
  if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
    throw std::runtime_error("RMSNorm epsilon must be a positive finite value");
  }

  double sum_squares = 0.0;
  for (const float value : input) {
    sum_squares += static_cast<double>(value) * static_cast<double>(value);
  }

  RmsNormOutput output;
  output.mean_square = sum_squares / static_cast<double>(input.size());
  output.inv_rms = 1.0 / std::sqrt(output.mean_square + epsilon);
  output.values.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const double value = static_cast<double>(input[i]) * output.inv_rms *
                         static_cast<double>(weights[i]);
    output.values.push_back(static_cast<float>(value));
  }
  return output;
}

struct VectorStats {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double l2_norm = 0.0;
  double checksum = 0.0;
};

VectorStats ComputeStats(const std::vector<float>& values) {
  if (values.empty()) {
    throw std::runtime_error("Cannot compute stats for an empty vector");
  }

  VectorStats stats;
  stats.min = values.front();
  stats.max = values.front();
  double sum = 0.0;
  double l2_sum = 0.0;
  double checksum = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    stats.min = std::min(stats.min, value);
    stats.max = std::max(stats.max, value);
    sum += value;
    l2_sum += value * value;
    checksum += value * static_cast<double>(i + 1);
  }
  stats.mean = sum / static_cast<double>(values.size());
  stats.l2_norm = std::sqrt(l2_sum);
  stats.checksum = checksum;
  return stats;
}

void ApplyRope(std::vector<float>& values, uint64_t position, uint64_t head_dim,
               double rope_theta) {
  if (values.empty()) {
    throw std::runtime_error("Cannot apply RoPE to an empty vector");
  }
  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::runtime_error("RoPE headDim must be a positive even value");
  }
  if (values.size() % static_cast<size_t>(head_dim) != 0) {
    throw std::runtime_error("RoPE output length must be divisible by headDim");
  }
  if (!(rope_theta > 0.0) || !std::isfinite(rope_theta)) {
    throw std::runtime_error("RoPE theta must be a positive finite value");
  }

  const size_t head_dim_size = static_cast<size_t>(head_dim);
  const size_t pair_count = head_dim_size / 2;
  for (size_t head_base = 0; head_base < values.size();
       head_base += head_dim_size) {
    for (size_t pair = 0; pair < pair_count; ++pair) {
      const double exponent =
          static_cast<double>(2 * pair) / static_cast<double>(head_dim_size);
      const double frequency = 1.0 / std::pow(rope_theta, exponent);
      const double angle = static_cast<double>(position) * frequency;
      const double cosine = std::cos(angle);
      const double sine = std::sin(angle);
      const size_t i = head_base + pair * 2;
      const double x0 = values[i];
      const double x1 = values[i + 1];
      values[i] = static_cast<float>(x0 * cosine - x1 * sine);
      values[i + 1] = static_cast<float>(x0 * sine + x1 * cosine);
    }
  }
}

std::vector<float> ProjectVectorRows(int fd, const TensorInfo& weight,
                                     const std::vector<float>& input) {
  if (weight.shape.size() < 2) {
    throw std::runtime_error("Projection weight tensor must be 2D");
  }
  const uint64_t input_length = weight.shape[0];
  const uint64_t output_length = weight.shape[1];
  if (input_length == 0 || output_length == 0) {
    throw std::runtime_error("Projection weight tensor has an invalid shape");
  }
  if (input.size() != CheckedSize(input_length, "Projection input too large")) {
    throw std::runtime_error("Projection input width does not match vector");
  }

  std::vector<float> output;
  output.reserve(CheckedSize(output_length, "Projection output too large"));
  for (uint64_t row = 0; row < output_length; ++row) {
    const std::vector<float> weight_row =
        DecodeTensorRow(fd, weight, row, input_length);
    double dot = 0.0;
    for (uint64_t col = 0; col < input_length; ++col) {
      dot += static_cast<double>(weight_row[static_cast<size_t>(col)]) *
             static_cast<double>(input[static_cast<size_t>(col)]);
    }
    output.push_back(static_cast<float>(dot));
  }
  return output;
}

std::vector<float> SliceHead(const std::vector<float>& values,
                             uint64_t head_dim, uint64_t head_index,
                             const std::string& label) {
  if (head_dim == 0) {
    throw std::runtime_error(label + " headDim must be positive");
  }
  if (values.size() % static_cast<size_t>(head_dim) != 0) {
    throw std::runtime_error(label + " width must be divisible by headDim");
  }
  const uint64_t head_count =
      static_cast<uint64_t>(values.size()) / head_dim;
  if (head_index >= head_count) {
    throw std::runtime_error(label + " headIndex is outside available heads");
  }
  const size_t start = CheckedSize(CheckedMultiply(head_index, head_dim),
                                   label + " head offset too large");
  const size_t length = CheckedSize(head_dim, label + " headDim too large");
  return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(start),
                            values.begin() +
                                static_cast<std::ptrdiff_t>(start + length));
}

std::vector<float> AddVectors(const std::vector<float>& left,
                              const std::vector<float>& right,
                              const std::string& label) {
  if (left.size() != right.size()) {
    throw std::runtime_error(label + " vector sizes do not match");
  }
  std::vector<float> output;
  output.reserve(left.size());
  for (size_t i = 0; i < left.size(); ++i) {
    output.push_back(static_cast<float>(static_cast<double>(left[i]) +
                                        static_cast<double>(right[i])));
  }
  return output;
}

float Silu(float value) {
  const double x = static_cast<double>(value);
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return static_cast<float>(x / (1.0 + z));
  }
  const double z = std::exp(x);
  return static_cast<float>((x * z) / (1.0 + z));
}

std::vector<float> ApplySwiGlu(const std::vector<float>& gate,
                               const std::vector<float>& up) {
  if (gate.size() != up.size()) {
    throw std::runtime_error("SwiGLU gate and up vector sizes do not match");
  }
  if (gate.empty()) {
    throw std::runtime_error("Cannot apply SwiGLU to an empty vector");
  }
  std::vector<float> output;
  output.reserve(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    output.push_back(static_cast<float>(static_cast<double>(Silu(gate[i])) *
                                        static_cast<double>(up[i])));
  }
  return output;
}

GgufIndex ReadGgufIndex(const std::string& path) {
  Reader reader(path);
  const uint32_t magic = reader.read_pod<uint32_t>();
  if (magic != kGgufMagic) {
    throw std::runtime_error("File is not a GGUF model");
  }

  GgufIndex index;
  index.path = path;
  index.file_size = reader.file_size();
  index.version = reader.read_pod<uint32_t>();
  index.tensor_count = reader.read_pod<uint64_t>();
  index.metadata_count = reader.read_pod<uint64_t>();

  for (uint64_t i = 0; i < index.metadata_count; ++i) {
    const std::string key = reader.read_string();
    const uint32_t type = reader.read_pod<uint32_t>();
    index.metadata[key] = ReadMetadataValue(reader, type);
  }

  index.tensors.reserve(
      static_cast<size_t>(std::min<uint64_t>(index.tensor_count, 4096)));
  for (uint64_t i = 0; i < index.tensor_count; ++i) {
    TensorInfo tensor;
    tensor.name = reader.read_string();
    const uint32_t dims = reader.read_pod<uint32_t>();
    if (dims > kMaxTensorDims) {
      throw std::runtime_error("Tensor has too many dimensions");
    }
    uint64_t elements = 1;
    for (uint32_t dim = 0; dim < dims; ++dim) {
      const uint64_t size = reader.read_pod<uint64_t>();
      tensor.shape.push_back(size);
      elements = CheckedMultiply(elements, size);
    }
    tensor.type = reader.read_pod<uint32_t>();
    tensor.relative_offset = reader.read_pod<uint64_t>();
    tensor.element_count = elements;
    tensor.has_known_byte_size = TensorDataBytes(tensor.type, elements,
                                                 tensor.byte_size);
    index.parameter_count += elements;
    index.tensor_types[TensorTypeName(tensor.type)] += 1;
    index.tensors.push_back(std::move(tensor));
  }

  index.alignment = ParseUnsigned(index.metadata, "general.alignment", 32);
  index.data_start = AlignOffset(reader.position(), index.alignment);
  index.architecture = FindMetadataString(index.metadata, "general.architecture");
  index.arch_prefix = index.architecture.empty() ? "" : index.architecture + ".";

  for (TensorInfo& tensor : index.tensors) {
    tensor.absolute_offset = CheckedAdd(index.data_start, tensor.relative_offset);
    if (tensor.has_known_byte_size) {
      const uint64_t end = CheckedAdd(tensor.absolute_offset, tensor.byte_size);
      if (end > index.file_size) {
        throw std::runtime_error("Tensor data extends beyond GGUF file end");
      }
    }
  }

  return index;
}

std::string InspectGguf(const std::string& path) {
  const GgufIndex index = ReadGgufIndex(path);

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonNumberField(out, "fileSizeBytes", index.file_size, first);
  AppendJsonNumberField(out, "version", index.version, first);
  AppendJsonNumberField(out, "tensorCount", index.tensor_count, first);
  AppendJsonNumberField(out, "metadataCount", index.metadata_count, first);
  AppendJsonNumberField(out, "alignment", index.alignment, first);
  AppendJsonNumberField(out, "dataStartOffset", index.data_start, first);
  AppendJsonNumberField(out, "parameterCount", index.parameter_count, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonField(out, "name", FindMetadataString(index.metadata, "general.name"),
                  first);
  AppendJsonField(out, "fileType",
                  FindMetadataString(index.metadata, "general.file_type"), first);
  AppendJsonField(out, "tokenizerModel",
                  FindMetadataString(index.metadata, "tokenizer.ggml.model"),
                  first);

  const std::vector<std::pair<std::string, std::string>> known_numbers = {
      {"blockCount", index.arch_prefix + "block_count"},
      {"contextLength", index.arch_prefix + "context_length"},
      {"embeddingLength", index.arch_prefix + "embedding_length"},
      {"feedForwardLength", index.arch_prefix + "feed_forward_length"},
      {"headCount", index.arch_prefix + "attention.head_count"},
      {"kvHeadCount", index.arch_prefix + "attention.head_count_kv"},
  };
  for (const auto& entry : known_numbers) {
    AppendJsonNumberField(out, entry.first,
                          ParseUnsigned(index.metadata, entry.second, 0), first);
  }
  AppendJsonField(out, "ropeFreqBase",
                  FindMetadataString(index.metadata,
                                     index.arch_prefix + "rope.freq_base"),
                  first);

  if (!first) out << ",";
  first = false;
  out << Quote("tensorTypes") << ":{";
  bool type_first = true;
  for (const auto& entry : index.tensor_types) {
    if (!type_first) out << ",";
    type_first = false;
    out << Quote(entry.first) << ":" << entry.second;
  }
  out << "}";

  out << "," << Quote("metadata") << ":{";
  bool metadata_first = true;
  for (const auto& entry : index.metadata) {
    if (!metadata_first) out << ",";
    metadata_first = false;
    out << Quote(entry.first) << ":" << Quote(entry.second);
  }
  out << "}";

  out << "," << Quote("tensors") << ":[";
  for (size_t i = 0; i < index.tensors.size(); ++i) {
    const TensorInfo& tensor = index.tensors[i];
    if (i != 0) out << ",";
    out << "{";
    bool tensor_first = true;
    AppendJsonField(out, "name", tensor.name, tensor_first);
    AppendJsonField(out, "type", TensorTypeName(tensor.type), tensor_first);
    AppendJsonNumberField(out, "typeId", tensor.type, tensor_first);
    AppendJsonNumberField(out, "relativeOffset", tensor.relative_offset,
                          tensor_first);
    AppendJsonNumberField(out, "absoluteOffset", tensor.absolute_offset,
                          tensor_first);
    AppendJsonNumberField(out, "elementCount", tensor.element_count,
                          tensor_first);
    AppendJsonBoolField(out, "hasKnownByteSize", tensor.has_known_byte_size,
                        tensor_first);
    AppendJsonNumberField(out, "byteSize", tensor.byte_size, tensor_first);
    AppendJsonSignedNumberField(out, "layerIndex",
                                LayerIndexFromTensorName(tensor.name),
                                tensor_first);
    if (!tensor_first) out << ",";
    out << Quote("shape") << ":[";
    for (size_t dim = 0; dim < tensor.shape.size(); ++dim) {
      if (dim != 0) out << ",";
      out << tensor.shape[dim];
    }
    out << "]";
    out << "}";
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string ReadTokenEmbedding(const std::string& path, int token_id,
                               int max_values) {
  if (token_id < 0) {
    throw std::runtime_error("Token id must be non-negative");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (!embedding->has_known_byte_size) {
    throw std::runtime_error(
        "token_embd.weight uses an unsupported GGML tensor type");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  if (embedding_length == 0 || vocab_size == 0) {
    throw std::runtime_error("token_embd.weight has an invalid shape");
  }
  if (static_cast<uint64_t>(token_id) >= vocab_size) {
    throw std::runtime_error("Token id is outside token_embd.weight vocab");
  }
  uint64_t row_byte_size = 0;
  if (!TensorRowBytes(embedding->type, embedding_length, row_byte_size)) {
    throw std::runtime_error("Unsupported token embedding tensor type: " +
                             TensorTypeName(embedding->type));
  }

  FileDescriptor fd(path);
  const std::vector<float> values =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                      embedding_length);
  if (values.size() != embedding_length) {
    throw std::runtime_error("Decoded embedding row has an unexpected length");
  }

  double min_value = values.front();
  double max_value = values.front();
  double sum = 0.0;
  double l2_sum = 0.0;
  double checksum = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    sum += value;
    l2_sum += value * value;
    checksum += value * static_cast<double>(i + 1);
  }

  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_values =
      std::min<size_t>(static_cast<size_t>(preview_count), values.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonField(out, "tensorName", embedding->name, first);
  AppendJsonField(out, "tensorType", TensorTypeName(embedding->type), first);
  AppendJsonNumberField(out, "tensorTypeId", embedding->type, first);
  AppendJsonNumberField(out, "tokenId", static_cast<uint64_t>(token_id), first);
  AppendJsonNumberField(out, "embeddingLength", embedding_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "rowByteSize", row_byte_size, first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedValueCount",
                        static_cast<uint64_t>(returned_values), first);
  AppendJsonFloatField(out, "min", min_value, first);
  AppendJsonFloatField(out, "max", max_value, first);
  AppendJsonFloatField(out, "mean", sum / static_cast<double>(values.size()),
                       first);
  AppendJsonFloatField(out, "l2Norm", std::sqrt(l2_sum), first);
  AppendJsonFloatField(out, "checksum", checksum, first);

  out << "," << Quote("shape") << ":[";
  for (size_t dim = 0; dim < embedding->shape.size(); ++dim) {
    if (dim != 0) out << ",";
    out << embedding->shape[dim];
  }
  out << "]";

  out << "," << Quote("values") << ":[";
  for (size_t i = 0; i < returned_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(values[i])) {
      out << std::setprecision(9) << values[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string RmsNormTokenEmbedding(const std::string& path, int token_id,
                                  const std::string& norm_tensor_name,
                                  float epsilon, int max_values) {
  if (token_id < 0) {
    throw std::runtime_error("Token id must be non-negative");
  }
  if (norm_tensor_name.empty()) {
    throw std::runtime_error("Norm tensor name must be non-empty");
  }
  if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
    throw std::runtime_error("RMSNorm epsilon must be a positive finite value");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  const TensorInfo* norm = FindTensor(index, norm_tensor_name);
  if (norm == nullptr) {
    throw std::runtime_error("GGUF norm tensor was not found: " +
                             norm_tensor_name);
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (norm->shape.empty()) {
    throw std::runtime_error("Norm tensor must be at least 1D");
  }
  if (!embedding->has_known_byte_size || !norm->has_known_byte_size) {
    throw std::runtime_error(
        "RMSNorm input tensors use unsupported GGML tensor types");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  if (embedding_length == 0 || vocab_size == 0) {
    throw std::runtime_error("token_embd.weight has an invalid shape");
  }
  if (norm->element_count != embedding_length) {
    throw std::runtime_error("Norm tensor length does not match embedding");
  }
  if (static_cast<uint64_t>(token_id) >= vocab_size) {
    throw std::runtime_error("Token id is outside token_embd.weight vocab");
  }

  FileDescriptor fd(path);
  const std::vector<float> input =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                      embedding_length);
  const std::vector<float> weights = DecodeTensorVector(fd.get(), *norm);
  if (input.size() != weights.size()) {
    throw std::runtime_error("Decoded RMSNorm vector sizes do not match");
  }

  double sum_squares = 0.0;
  for (const float value : input) {
    sum_squares += static_cast<double>(value) * static_cast<double>(value);
  }
  const double mean_square = sum_squares / static_cast<double>(input.size());
  const double inv_rms = 1.0 / std::sqrt(mean_square + epsilon);

  std::vector<float> output;
  output.reserve(input.size());
  double min_value = 0.0;
  double max_value = 0.0;
  double sum = 0.0;
  double l2_sum = 0.0;
  double checksum = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    const double value =
        static_cast<double>(input[i]) * inv_rms * static_cast<double>(weights[i]);
    output.push_back(static_cast<float>(value));
    if (i == 0) {
      min_value = value;
      max_value = value;
    } else {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    sum += value;
    l2_sum += value * value;
    checksum += value * static_cast<double>(i + 1);
  }

  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_values =
      std::min<size_t>(static_cast<size_t>(preview_count), output.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "tokenId", static_cast<uint64_t>(token_id), first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "normTensorName", norm->name, first);
  AppendJsonField(out, "normTensorType", TensorTypeName(norm->type), first);
  AppendJsonNumberField(out, "embeddingLength", embedding_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedValueCount",
                        static_cast<uint64_t>(returned_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "meanSquare", mean_square, first);
  AppendJsonFloatField(out, "invRms", inv_rms, first);
  AppendJsonFloatField(out, "min", min_value, first);
  AppendJsonFloatField(out, "max", max_value, first);
  AppendJsonFloatField(out, "mean", sum / static_cast<double>(output.size()),
                       first);
  AppendJsonFloatField(out, "l2Norm", std::sqrt(l2_sum), first);
  AppendJsonFloatField(out, "checksum", checksum, first);

  out << "," << Quote("values") << ":[";
  for (size_t i = 0; i < returned_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(output[i])) {
      out << std::setprecision(9) << output[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string MatVecRmsNormTokenEmbedding(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    const std::string& weight_tensor_name, float epsilon, int max_values) {
  if (token_id < 0) {
    throw std::runtime_error("Token id must be non-negative");
  }
  if (norm_tensor_name.empty()) {
    throw std::runtime_error("Norm tensor name must be non-empty");
  }
  if (weight_tensor_name.empty()) {
    throw std::runtime_error("Weight tensor name must be non-empty");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  const TensorInfo* norm = FindTensor(index, norm_tensor_name);
  const TensorInfo* weight = FindTensor(index, weight_tensor_name);
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (norm == nullptr) {
    throw std::runtime_error("GGUF norm tensor was not found: " +
                             norm_tensor_name);
  }
  if (weight == nullptr) {
    throw std::runtime_error("GGUF weight tensor was not found: " +
                             weight_tensor_name);
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (weight->shape.size() < 2) {
    throw std::runtime_error("Projection weight tensor must be 2D");
  }
  if (!embedding->has_known_byte_size || !norm->has_known_byte_size ||
      !weight->has_known_byte_size) {
    throw std::runtime_error(
        "MatVec input tensors use unsupported GGML tensor types");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  const uint64_t input_length = weight->shape[0];
  const uint64_t output_length = weight->shape[1];
  if (embedding_length == 0 || vocab_size == 0 || input_length == 0 ||
      output_length == 0) {
    throw std::runtime_error("MatVec tensor has an invalid shape");
  }
  if (norm->element_count != embedding_length) {
    throw std::runtime_error("Norm tensor length does not match embedding");
  }
  if (input_length != embedding_length) {
    throw std::runtime_error("Projection input width does not match embedding");
  }
  if (static_cast<uint64_t>(token_id) >= vocab_size) {
    throw std::runtime_error("Token id is outside token_embd.weight vocab");
  }

  FileDescriptor fd(path);
  const std::vector<float> input =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                      embedding_length);
  const std::vector<float> norm_weights = DecodeTensorVector(fd.get(), *norm);
  const RmsNormOutput normalized = ApplyRmsNorm(input, norm_weights, epsilon);

  std::vector<float> output;
  output.reserve(static_cast<size_t>(output_length));
  for (uint64_t row = 0; row < output_length; ++row) {
    const std::vector<float> weight_row =
        DecodeTensorRow(fd.get(), *weight, row, input_length);
    double dot = 0.0;
    for (uint64_t col = 0; col < input_length; ++col) {
      dot += static_cast<double>(weight_row[static_cast<size_t>(col)]) *
             static_cast<double>(normalized.values[static_cast<size_t>(col)]);
    }
    output.push_back(static_cast<float>(dot));
  }

  const VectorStats stats = ComputeStats(output);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_values =
      std::min<size_t>(static_cast<size_t>(preview_count), output.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "tokenId", static_cast<uint64_t>(token_id), first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "normTensorName", norm->name, first);
  AppendJsonField(out, "normTensorType", TensorTypeName(norm->type), first);
  AppendJsonField(out, "weightTensorName", weight->name, first);
  AppendJsonField(out, "weightTensorType", TensorTypeName(weight->type), first);
  AppendJsonNumberField(out, "inputLength", input_length, first);
  AppendJsonNumberField(out, "outputLength", output_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedValueCount",
                        static_cast<uint64_t>(returned_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "meanSquare", normalized.mean_square, first);
  AppendJsonFloatField(out, "invRms", normalized.inv_rms, first);
  AppendJsonFloatField(out, "min", stats.min, first);
  AppendJsonFloatField(out, "max", stats.max, first);
  AppendJsonFloatField(out, "mean", stats.mean, first);
  AppendJsonFloatField(out, "l2Norm", stats.l2_norm, first);
  AppendJsonFloatField(out, "checksum", stats.checksum, first);

  out << "," << Quote("weightShape") << ":[";
  for (size_t dim = 0; dim < weight->shape.size(); ++dim) {
    if (dim != 0) out << ",";
    out << weight->shape[dim];
  }
  out << "]";

  out << "," << Quote("values") << ":[";
  for (size_t i = 0; i < returned_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(output[i])) {
      out << std::setprecision(9) << output[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string RopeMatVecRmsNormTokenEmbedding(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    const std::string& weight_tensor_name, float epsilon, int64_t position,
    int head_dim, double rope_theta, int max_values) {
  if (token_id < 0) {
    throw std::runtime_error("Token id must be non-negative");
  }
  if (position < 0) {
    throw std::runtime_error("RoPE position must be non-negative");
  }
  if (head_dim <= 0) {
    throw std::runtime_error("RoPE headDim must be positive");
  }
  if (norm_tensor_name.empty()) {
    throw std::runtime_error("Norm tensor name must be non-empty");
  }
  if (weight_tensor_name.empty()) {
    throw std::runtime_error("Weight tensor name must be non-empty");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  const TensorInfo* norm = FindTensor(index, norm_tensor_name);
  const TensorInfo* weight = FindTensor(index, weight_tensor_name);
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (norm == nullptr) {
    throw std::runtime_error("GGUF norm tensor was not found: " +
                             norm_tensor_name);
  }
  if (weight == nullptr) {
    throw std::runtime_error("GGUF weight tensor was not found: " +
                             weight_tensor_name);
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (weight->shape.size() < 2) {
    throw std::runtime_error("Projection weight tensor must be 2D");
  }
  if (!embedding->has_known_byte_size || !norm->has_known_byte_size ||
      !weight->has_known_byte_size) {
    throw std::runtime_error(
        "RoPE input tensors use unsupported GGML tensor types");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  const uint64_t input_length = weight->shape[0];
  const uint64_t output_length = weight->shape[1];
  const uint64_t head_dim_u64 = static_cast<uint64_t>(head_dim);
  if (embedding_length == 0 || vocab_size == 0 || input_length == 0 ||
      output_length == 0) {
    throw std::runtime_error("RoPE MatVec tensor has an invalid shape");
  }
  if (norm->element_count != embedding_length) {
    throw std::runtime_error("Norm tensor length does not match embedding");
  }
  if (input_length != embedding_length) {
    throw std::runtime_error("Projection input width does not match embedding");
  }
  if (output_length % head_dim_u64 != 0) {
    throw std::runtime_error("RoPE output length must be divisible by headDim");
  }
  if (static_cast<uint64_t>(token_id) >= vocab_size) {
    throw std::runtime_error("Token id is outside token_embd.weight vocab");
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  FileDescriptor fd(path);
  const std::vector<float> input =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                      embedding_length);
  const std::vector<float> norm_weights = DecodeTensorVector(fd.get(), *norm);
  const RmsNormOutput normalized = ApplyRmsNorm(input, norm_weights, epsilon);

  std::vector<float> output;
  output.reserve(static_cast<size_t>(output_length));
  for (uint64_t row = 0; row < output_length; ++row) {
    const std::vector<float> weight_row =
        DecodeTensorRow(fd.get(), *weight, row, input_length);
    double dot = 0.0;
    for (uint64_t col = 0; col < input_length; ++col) {
      dot += static_cast<double>(weight_row[static_cast<size_t>(col)]) *
             static_cast<double>(normalized.values[static_cast<size_t>(col)]);
    }
    output.push_back(static_cast<float>(dot));
  }

  ApplyRope(output, static_cast<uint64_t>(position), head_dim_u64,
            resolved_rope_theta);

  const VectorStats stats = ComputeStats(output);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_values =
      std::min<size_t>(static_cast<size_t>(preview_count), output.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "tokenId", static_cast<uint64_t>(token_id), first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "normTensorName", norm->name, first);
  AppendJsonField(out, "normTensorType", TensorTypeName(norm->type), first);
  AppendJsonField(out, "weightTensorName", weight->name, first);
  AppendJsonField(out, "weightTensorType", TensorTypeName(weight->type), first);
  AppendJsonNumberField(out, "inputLength", input_length, first);
  AppendJsonNumberField(out, "outputLength", output_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "position", static_cast<uint64_t>(position),
                        first);
  AppendJsonNumberField(out, "headDim", head_dim_u64, first);
  AppendJsonNumberField(out, "headCount", output_length / head_dim_u64, first);
  AppendJsonNumberField(out, "rotatedPairCount", output_length / 2, first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedValueCount",
                        static_cast<uint64_t>(returned_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "meanSquare", normalized.mean_square, first);
  AppendJsonFloatField(out, "invRms", normalized.inv_rms, first);
  AppendJsonFloatField(out, "ropeTheta", resolved_rope_theta, first);
  AppendJsonFloatField(out, "min", stats.min, first);
  AppendJsonFloatField(out, "max", stats.max, first);
  AppendJsonFloatField(out, "mean", stats.mean, first);
  AppendJsonFloatField(out, "l2Norm", stats.l2_norm, first);
  AppendJsonFloatField(out, "checksum", stats.checksum, first);

  out << "," << Quote("weightShape") << ":[";
  for (size_t dim = 0; dim < weight->shape.size(); ++dim) {
    if (dim != 0) out << ",";
    out << weight->shape[dim];
  }
  out << "]";

  out << "," << Quote("values") << ":[";
  for (size_t i = 0; i < returned_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(output[i])) {
      out << std::setprecision(9) << output[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string KvCacheRmsNormTokenEmbeddings(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& norm_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t read_position, int head_dim,
    double rope_theta, int max_values) {
  if (token_ids.empty()) {
    throw std::runtime_error("KV cache tokenIds must be non-empty");
  }
  if (token_ids.size() > 4096) {
    throw std::runtime_error("KV cache diagnostic sequence is too large");
  }
  if (start_position < 0) {
    throw std::runtime_error("KV cache startPosition must be non-negative");
  }
  if (head_dim <= 0) {
    throw std::runtime_error("KV cache headDim must be positive");
  }
  if (norm_tensor_name.empty()) {
    throw std::runtime_error("Norm tensor name must be non-empty");
  }
  if (key_weight_tensor_name.empty() || value_weight_tensor_name.empty()) {
    throw std::runtime_error("KV cache key/value tensor names must be non-empty");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error("Token ids must be non-negative");
    }
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  const TensorInfo* norm = FindTensor(index, norm_tensor_name);
  const TensorInfo* key_weight = FindTensor(index, key_weight_tensor_name);
  const TensorInfo* value_weight = FindTensor(index, value_weight_tensor_name);
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (norm == nullptr) {
    throw std::runtime_error("GGUF norm tensor was not found: " +
                             norm_tensor_name);
  }
  if (key_weight == nullptr) {
    throw std::runtime_error("GGUF key weight tensor was not found: " +
                             key_weight_tensor_name);
  }
  if (value_weight == nullptr) {
    throw std::runtime_error("GGUF value weight tensor was not found: " +
                             value_weight_tensor_name);
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (key_weight->shape.size() < 2 || value_weight->shape.size() < 2) {
    throw std::runtime_error("KV projection weight tensors must be 2D");
  }
  if (!embedding->has_known_byte_size || !norm->has_known_byte_size ||
      !key_weight->has_known_byte_size || !value_weight->has_known_byte_size) {
    throw std::runtime_error(
        "KV cache input tensors use unsupported GGML tensor types");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  const uint64_t key_input_length = key_weight->shape[0];
  const uint64_t key_width = key_weight->shape[1];
  const uint64_t value_input_length = value_weight->shape[0];
  const uint64_t value_width = value_weight->shape[1];
  const uint64_t head_dim_u64 = static_cast<uint64_t>(head_dim);
  if (embedding_length == 0 || vocab_size == 0 || key_input_length == 0 ||
      key_width == 0 || value_input_length == 0 || value_width == 0) {
    throw std::runtime_error("KV cache tensors have an invalid shape");
  }
  if (norm->element_count != embedding_length) {
    throw std::runtime_error("Norm tensor length does not match embedding");
  }
  if (key_input_length != embedding_length ||
      value_input_length != embedding_length) {
    throw std::runtime_error("KV projection input width does not match embedding");
  }
  if (key_width != value_width) {
    throw std::runtime_error("KV cache key and value widths must match");
  }
  if (key_width % head_dim_u64 != 0) {
    throw std::runtime_error("KV cache width must be divisible by headDim");
  }
  for (const int token_id : token_ids) {
    if (static_cast<uint64_t>(token_id) >= vocab_size) {
      throw std::runtime_error("Token id is outside token_embd.weight vocab");
    }
  }

  int64_t read_index = read_position;
  if (read_index < 0) {
    read_index = static_cast<int64_t>(token_ids.size() - 1);
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >=
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error("KV cache readPosition is outside the sequence");
  }
  if (start_position >
      std::numeric_limits<int64_t>::max() -
          static_cast<int64_t>(token_ids.size() - 1)) {
    throw std::runtime_error("KV cache positions overflow");
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  const uint64_t sequence_length = static_cast<uint64_t>(token_ids.size());
  const uint64_t cache_element_count = CheckedMultiply(
      sequence_length, CheckedAdd(key_width, value_width));
  const uint64_t cache_bytes_fp32 =
      CheckedMultiply(cache_element_count, static_cast<uint64_t>(sizeof(float)));

  FileDescriptor fd(path);
  const std::vector<float> norm_weights = DecodeTensorVector(fd.get(), *norm);
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  key_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, key_width),
                                "KV key cache too large"));
  value_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, value_width),
                                  "KV value cache too large"));

  double last_mean_square = 0.0;
  double last_inv_rms = 0.0;
  for (size_t index_in_sequence = 0; index_in_sequence < token_ids.size();
       ++index_in_sequence) {
    const int token_id = token_ids[index_in_sequence];
    const std::vector<float> input =
        DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                        embedding_length);
    const RmsNormOutput normalized = ApplyRmsNorm(input, norm_weights, epsilon);
    last_mean_square = normalized.mean_square;
    last_inv_rms = normalized.inv_rms;

    std::vector<float> key =
        ProjectVectorRows(fd.get(), *key_weight, normalized.values);
    std::vector<float> value =
        ProjectVectorRows(fd.get(), *value_weight, normalized.values);
    const int64_t absolute_position =
        start_position + static_cast<int64_t>(index_in_sequence);
    ApplyRope(key, static_cast<uint64_t>(absolute_position), head_dim_u64,
              resolved_rope_theta);
    key_cache.insert(key_cache.end(), key.begin(), key.end());
    value_cache.insert(value_cache.end(), value.begin(), value.end());
  }

  const VectorStats key_stats = ComputeStats(key_cache);
  const VectorStats value_stats = ComputeStats(value_cache);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_key_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       CheckedSize(key_width, "KV key width too large"));
  const size_t returned_value_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       CheckedSize(value_width, "KV value width too large"));
  const size_t read_key_offset =
      CheckedSize(CheckedMultiply(static_cast<uint64_t>(read_index), key_width),
                  "KV key read offset too large");
  const size_t read_value_offset = CheckedSize(
      CheckedMultiply(static_cast<uint64_t>(read_index), value_width),
      "KV value read offset too large");
  const int64_t read_absolute_position = start_position + read_index;

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "normTensorName", norm->name, first);
  AppendJsonField(out, "normTensorType", TensorTypeName(norm->type), first);
  AppendJsonField(out, "keyWeightTensorName", key_weight->name, first);
  AppendJsonField(out, "keyWeightTensorType", TensorTypeName(key_weight->type),
                  first);
  AppendJsonField(out, "valueWeightTensorName", value_weight->name, first);
  AppendJsonField(out, "valueWeightTensorType",
                  TensorTypeName(value_weight->type), first);
  AppendJsonNumberField(out, "sequenceLength", sequence_length, first);
  AppendJsonNumberField(out, "embeddingLength", embedding_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "keyWidth", key_width, first);
  AppendJsonNumberField(out, "valueWidth", value_width, first);
  AppendJsonNumberField(out, "headDim", head_dim_u64, first);
  AppendJsonNumberField(out, "kvHeadCount", key_width / head_dim_u64, first);
  AppendJsonNumberField(out, "cacheElementCount", cache_element_count, first);
  AppendJsonNumberField(out, "cacheBytesFp32", cache_bytes_fp32, first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(read_index), first);
  AppendJsonNumberField(out, "readAbsolutePosition",
                        static_cast<uint64_t>(read_absolute_position), first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedKeyValueCount",
                        static_cast<uint64_t>(returned_key_values), first);
  AppendJsonNumberField(out, "returnedValueValueCount",
                        static_cast<uint64_t>(returned_value_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "lastMeanSquare", last_mean_square, first);
  AppendJsonFloatField(out, "lastInvRms", last_inv_rms, first);
  AppendJsonFloatField(out, "ropeTheta", resolved_rope_theta, first);
  AppendJsonFloatField(out, "keyMin", key_stats.min, first);
  AppendJsonFloatField(out, "keyMax", key_stats.max, first);
  AppendJsonFloatField(out, "keyMean", key_stats.mean, first);
  AppendJsonFloatField(out, "keyL2Norm", key_stats.l2_norm, first);
  AppendJsonFloatField(out, "keyChecksum", key_stats.checksum, first);
  AppendJsonFloatField(out, "valueMin", value_stats.min, first);
  AppendJsonFloatField(out, "valueMax", value_stats.max, first);
  AppendJsonFloatField(out, "valueMean", value_stats.mean, first);
  AppendJsonFloatField(out, "valueL2Norm", value_stats.l2_norm, first);
  AppendJsonFloatField(out, "valueChecksum", value_stats.checksum, first);

  out << "," << Quote("tokenIds") << ":[";
  for (size_t i = 0; i < token_ids.size(); ++i) {
    if (i != 0) out << ",";
    out << token_ids[i];
  }
  out << "]";

  out << "," << Quote("keyWeightShape") << ":[";
  for (size_t dim = 0; dim < key_weight->shape.size(); ++dim) {
    if (dim != 0) out << ",";
    out << key_weight->shape[dim];
  }
  out << "]";

  out << "," << Quote("valueWeightShape") << ":[";
  for (size_t dim = 0; dim < value_weight->shape.size(); ++dim) {
    if (dim != 0) out << ",";
    out << value_weight->shape[dim];
  }
  out << "]";

  out << "," << Quote("keyValues") << ":[";
  for (size_t i = 0; i < returned_key_values; ++i) {
    if (i != 0) out << ",";
    const float value = key_cache[read_key_offset + i];
    if (std::isfinite(value)) {
      out << std::setprecision(9) << value;
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("valueValues") << ":[";
  for (size_t i = 0; i < returned_value_values; ++i) {
    if (i != 0) out << ",";
    const float value = value_cache[read_value_offset + i];
    if (std::isfinite(value)) {
      out << std::setprecision(9) << value;
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string AttentionSingleHead(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, int head_index, double rope_theta,
    int max_values) {
  if (token_ids.empty()) {
    throw std::runtime_error("Attention tokenIds must be non-empty");
  }
  if (token_ids.size() > 4096) {
    throw std::runtime_error("Attention diagnostic sequence is too large");
  }
  if (query_token_id < 0) {
    throw std::runtime_error("Attention queryTokenId must be non-negative");
  }
  if (start_position < 0 || query_position < 0) {
    throw std::runtime_error("Attention positions must be non-negative");
  }
  if (head_dim <= 0 || head_index < 0) {
    throw std::runtime_error("Attention headDim/headIndex must be non-negative");
  }
  if (norm_tensor_name.empty() || query_weight_tensor_name.empty() ||
      key_weight_tensor_name.empty() || value_weight_tensor_name.empty()) {
    throw std::runtime_error("Attention tensor names must be non-empty");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error("Attention token ids must be non-negative");
    }
  }

  int read_index = read_position;
  if (read_index < 0) {
    read_index = static_cast<int>(token_ids.size() - 1);
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >=
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error("Attention readPosition is outside the sequence");
  }
  int resolved_attention_length = attention_length;
  if (resolved_attention_length <= 0) {
    resolved_attention_length = read_index + 1;
  }
  if (resolved_attention_length <= 0 ||
      static_cast<uint64_t>(resolved_attention_length) >
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error("Attention length is outside the sequence");
  }
  if (start_position >
      std::numeric_limits<int64_t>::max() -
          static_cast<int64_t>(token_ids.size() - 1)) {
    throw std::runtime_error("Attention cache positions overflow");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  const TensorInfo* norm = FindTensor(index, norm_tensor_name);
  const TensorInfo* query_weight = FindTensor(index, query_weight_tensor_name);
  const TensorInfo* key_weight = FindTensor(index, key_weight_tensor_name);
  const TensorInfo* value_weight = FindTensor(index, value_weight_tensor_name);
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (norm == nullptr) {
    throw std::runtime_error("GGUF norm tensor was not found: " +
                             norm_tensor_name);
  }
  if (query_weight == nullptr) {
    throw std::runtime_error("GGUF query weight tensor was not found: " +
                             query_weight_tensor_name);
  }
  if (key_weight == nullptr) {
    throw std::runtime_error("GGUF key weight tensor was not found: " +
                             key_weight_tensor_name);
  }
  if (value_weight == nullptr) {
    throw std::runtime_error("GGUF value weight tensor was not found: " +
                             value_weight_tensor_name);
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (query_weight->shape.size() < 2 || key_weight->shape.size() < 2 ||
      value_weight->shape.size() < 2) {
    throw std::runtime_error("Attention projection weight tensors must be 2D");
  }
  if (!embedding->has_known_byte_size || !norm->has_known_byte_size ||
      !query_weight->has_known_byte_size || !key_weight->has_known_byte_size ||
      !value_weight->has_known_byte_size) {
    throw std::runtime_error(
        "Attention input tensors use unsupported GGML tensor types");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  const uint64_t query_input_length = query_weight->shape[0];
  const uint64_t query_width = query_weight->shape[1];
  const uint64_t key_input_length = key_weight->shape[0];
  const uint64_t key_width = key_weight->shape[1];
  const uint64_t value_input_length = value_weight->shape[0];
  const uint64_t value_width = value_weight->shape[1];
  const uint64_t head_dim_u64 = static_cast<uint64_t>(head_dim);
  const uint64_t head_index_u64 = static_cast<uint64_t>(head_index);
  if (embedding_length == 0 || vocab_size == 0 || query_input_length == 0 ||
      query_width == 0 || key_input_length == 0 || key_width == 0 ||
      value_input_length == 0 || value_width == 0) {
    throw std::runtime_error("Attention tensors have an invalid shape");
  }
  if (norm->element_count != embedding_length) {
    throw std::runtime_error("Norm tensor length does not match embedding");
  }
  if (query_input_length != embedding_length ||
      key_input_length != embedding_length ||
      value_input_length != embedding_length) {
    throw std::runtime_error(
        "Attention projection input width does not match embedding");
  }
  if (key_width != value_width) {
    throw std::runtime_error("Attention key and value widths must match");
  }
  if (query_width % head_dim_u64 != 0 || key_width % head_dim_u64 != 0) {
    throw std::runtime_error(
        "Attention projection widths must be divisible by headDim");
  }
  if (head_index_u64 >= query_width / head_dim_u64 ||
      head_index_u64 >= key_width / head_dim_u64) {
    throw std::runtime_error("Attention headIndex is outside available heads");
  }
  if (static_cast<uint64_t>(query_token_id) >= vocab_size) {
    throw std::runtime_error("Query token id is outside token_embd.weight vocab");
  }
  for (const int token_id : token_ids) {
    if (static_cast<uint64_t>(token_id) >= vocab_size) {
      throw std::runtime_error("Token id is outside token_embd.weight vocab");
    }
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  FileDescriptor fd(path);
  const std::vector<float> norm_weights = DecodeTensorVector(fd.get(), *norm);

  std::vector<float> key_cache;
  std::vector<float> value_cache;
  const uint64_t sequence_length = static_cast<uint64_t>(token_ids.size());
  key_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, key_width),
                                "Attention key cache too large"));
  value_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, value_width),
                                  "Attention value cache too large"));

  for (size_t index_in_sequence = 0; index_in_sequence < token_ids.size();
       ++index_in_sequence) {
    const int token_id = token_ids[index_in_sequence];
    const std::vector<float> input =
        DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                        embedding_length);
    const RmsNormOutput normalized = ApplyRmsNorm(input, norm_weights, epsilon);
    std::vector<float> key =
        ProjectVectorRows(fd.get(), *key_weight, normalized.values);
    std::vector<float> value =
        ProjectVectorRows(fd.get(), *value_weight, normalized.values);
    const int64_t absolute_position =
        start_position + static_cast<int64_t>(index_in_sequence);
    ApplyRope(key, static_cast<uint64_t>(absolute_position), head_dim_u64,
              resolved_rope_theta);
    key_cache.insert(key_cache.end(), key.begin(), key.end());
    value_cache.insert(value_cache.end(), value.begin(), value.end());
  }

  const std::vector<float> query_input =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(query_token_id),
                      embedding_length);
  const RmsNormOutput query_normalized =
      ApplyRmsNorm(query_input, norm_weights, epsilon);
  std::vector<float> query_full =
      ProjectVectorRows(fd.get(), *query_weight, query_normalized.values);
  ApplyRope(query_full, static_cast<uint64_t>(query_position), head_dim_u64,
            resolved_rope_theta);
  const std::vector<float> query =
      SliceHead(query_full, head_dim_u64, head_index_u64, "Attention query");

  const size_t head_offset =
      CheckedSize(CheckedMultiply(head_index_u64, head_dim_u64),
                  "Attention head offset too large");
  const size_t head_dim_size =
      CheckedSize(head_dim_u64, "Attention headDim too large");

  std::vector<double> scores;
  scores.reserve(static_cast<size_t>(resolved_attention_length));
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_u64));
  double max_score = -std::numeric_limits<double>::infinity();
  for (int token_index = 0; token_index < resolved_attention_length;
       ++token_index) {
    const size_t key_base =
        CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                    key_width),
                    "Attention key offset too large") +
        head_offset;
    double dot = 0.0;
    for (size_t dim = 0; dim < head_dim_size; ++dim) {
      dot += static_cast<double>(query[dim]) *
             static_cast<double>(key_cache[key_base + dim]);
    }
    const double score = dot * scale;
    scores.push_back(score);
    max_score = std::max(max_score, score);
  }

  std::vector<double> probabilities;
  probabilities.reserve(scores.size());
  double sum_exp = 0.0;
  for (const double score : scores) {
    const double value = std::exp(score - max_score);
    probabilities.push_back(value);
    sum_exp += value;
  }
  if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
    throw std::runtime_error("Attention softmax produced an invalid sum");
  }
  for (double& probability : probabilities) {
    probability /= sum_exp;
  }

  std::vector<float> output(head_dim_size, 0.0f);
  for (int token_index = 0; token_index < resolved_attention_length;
       ++token_index) {
    const size_t value_base =
        CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                    value_width),
                    "Attention value offset too large") +
        head_offset;
    const double probability = probabilities[static_cast<size_t>(token_index)];
    for (size_t dim = 0; dim < head_dim_size; ++dim) {
      output[dim] = static_cast<float>(
          static_cast<double>(output[dim]) +
          probability * static_cast<double>(value_cache[value_base + dim]));
    }
  }

  const VectorStats output_stats = ComputeStats(output);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_output_values =
      std::min<size_t>(static_cast<size_t>(preview_count), output.size());
  const size_t returned_query_values =
      std::min<size_t>(static_cast<size_t>(preview_count), query.size());
  const size_t returned_score_values =
      std::min<size_t>(static_cast<size_t>(preview_count), scores.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "queryTokenId",
                        static_cast<uint64_t>(query_token_id), first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "normTensorName", norm->name, first);
  AppendJsonField(out, "normTensorType", TensorTypeName(norm->type), first);
  AppendJsonField(out, "queryWeightTensorName", query_weight->name, first);
  AppendJsonField(out, "queryWeightTensorType",
                  TensorTypeName(query_weight->type), first);
  AppendJsonField(out, "keyWeightTensorName", key_weight->name, first);
  AppendJsonField(out, "keyWeightTensorType", TensorTypeName(key_weight->type),
                  first);
  AppendJsonField(out, "valueWeightTensorName", value_weight->name, first);
  AppendJsonField(out, "valueWeightTensorType",
                  TensorTypeName(value_weight->type), first);
  AppendJsonNumberField(out, "sequenceLength", sequence_length, first);
  AppendJsonNumberField(out, "attentionLength",
                        static_cast<uint64_t>(resolved_attention_length),
                        first);
  AppendJsonNumberField(out, "embeddingLength", embedding_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "queryWidth", query_width, first);
  AppendJsonNumberField(out, "keyWidth", key_width, first);
  AppendJsonNumberField(out, "valueWidth", value_width, first);
  AppendJsonNumberField(out, "headDim", head_dim_u64, first);
  AppendJsonNumberField(out, "headIndex", head_index_u64, first);
  AppendJsonNumberField(out, "queryHeadCount", query_width / head_dim_u64,
                        first);
  AppendJsonNumberField(out, "kvHeadCount", key_width / head_dim_u64, first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "queryPosition",
                        static_cast<uint64_t>(query_position), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(read_index), first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedOutputValueCount",
                        static_cast<uint64_t>(returned_output_values), first);
  AppendJsonNumberField(out, "returnedScoreValueCount",
                        static_cast<uint64_t>(returned_score_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "queryMeanSquare", query_normalized.mean_square,
                       first);
  AppendJsonFloatField(out, "queryInvRms", query_normalized.inv_rms, first);
  AppendJsonFloatField(out, "ropeTheta", resolved_rope_theta, first);
  AppendJsonFloatField(out, "scale", scale, first);
  AppendJsonFloatField(out, "maxScore", max_score, first);
  AppendJsonFloatField(out, "softmaxDenominator", sum_exp, first);
  AppendJsonFloatField(out, "outputMin", output_stats.min, first);
  AppendJsonFloatField(out, "outputMax", output_stats.max, first);
  AppendJsonFloatField(out, "outputMean", output_stats.mean, first);
  AppendJsonFloatField(out, "outputL2Norm", output_stats.l2_norm, first);
  AppendJsonFloatField(out, "outputChecksum", output_stats.checksum, first);

  out << "," << Quote("tokenIds") << ":[";
  for (size_t i = 0; i < token_ids.size(); ++i) {
    if (i != 0) out << ",";
    out << token_ids[i];
  }
  out << "]";

  out << "," << Quote("queryValues") << ":[";
  for (size_t i = 0; i < returned_query_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(query[i])) {
      out << std::setprecision(9) << query[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("attentionScores") << ":[";
  for (size_t i = 0; i < returned_score_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(scores[i])) {
      out << std::setprecision(9) << scores[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("attentionProbabilities") << ":[";
  for (size_t i = 0; i < returned_score_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(probabilities[i])) {
      out << std::setprecision(9) << probabilities[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("outputValues") << ":[";
  for (size_t i = 0; i < returned_output_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(output[i])) {
      out << std::setprecision(9) << output[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string AttentionMultiHead(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, double rope_theta, int max_values) {
  if (token_ids.empty()) {
    throw std::runtime_error("Multi-head attention tokenIds must be non-empty");
  }
  if (token_ids.size() > 4096) {
    throw std::runtime_error(
        "Multi-head attention diagnostic sequence is too large");
  }
  if (query_token_id < 0) {
    throw std::runtime_error(
        "Multi-head attention queryTokenId must be non-negative");
  }
  if (start_position < 0 || query_position < 0) {
    throw std::runtime_error(
        "Multi-head attention positions must be non-negative");
  }
  if (head_dim <= 0) {
    throw std::runtime_error("Multi-head attention headDim must be positive");
  }
  if (norm_tensor_name.empty() || query_weight_tensor_name.empty() ||
      key_weight_tensor_name.empty() || value_weight_tensor_name.empty()) {
    throw std::runtime_error(
        "Multi-head attention tensor names must be non-empty");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error(
          "Multi-head attention token ids must be non-negative");
    }
  }

  int read_index = read_position;
  if (read_index < 0) {
    read_index = static_cast<int>(token_ids.size() - 1);
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >=
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error(
        "Multi-head attention readPosition is outside the sequence");
  }
  int resolved_attention_length = attention_length;
  if (resolved_attention_length <= 0) {
    resolved_attention_length = read_index + 1;
  }
  if (resolved_attention_length <= 0 ||
      static_cast<uint64_t>(resolved_attention_length) >
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error(
        "Multi-head attention length is outside the sequence");
  }
  if (start_position >
      std::numeric_limits<int64_t>::max() -
          static_cast<int64_t>(token_ids.size() - 1)) {
    throw std::runtime_error("Multi-head attention cache positions overflow");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  const TensorInfo* norm = FindTensor(index, norm_tensor_name);
  const TensorInfo* query_weight = FindTensor(index, query_weight_tensor_name);
  const TensorInfo* key_weight = FindTensor(index, key_weight_tensor_name);
  const TensorInfo* value_weight = FindTensor(index, value_weight_tensor_name);
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (norm == nullptr) {
    throw std::runtime_error("GGUF norm tensor was not found: " +
                             norm_tensor_name);
  }
  if (query_weight == nullptr) {
    throw std::runtime_error("GGUF query weight tensor was not found: " +
                             query_weight_tensor_name);
  }
  if (key_weight == nullptr) {
    throw std::runtime_error("GGUF key weight tensor was not found: " +
                             key_weight_tensor_name);
  }
  if (value_weight == nullptr) {
    throw std::runtime_error("GGUF value weight tensor was not found: " +
                             value_weight_tensor_name);
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (query_weight->shape.size() < 2 || key_weight->shape.size() < 2 ||
      value_weight->shape.size() < 2) {
    throw std::runtime_error(
        "Multi-head attention projection weight tensors must be 2D");
  }
  if (!embedding->has_known_byte_size || !norm->has_known_byte_size ||
      !query_weight->has_known_byte_size || !key_weight->has_known_byte_size ||
      !value_weight->has_known_byte_size) {
    throw std::runtime_error(
        "Multi-head attention input tensors use unsupported GGML tensor types");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  const uint64_t query_input_length = query_weight->shape[0];
  const uint64_t query_width = query_weight->shape[1];
  const uint64_t key_input_length = key_weight->shape[0];
  const uint64_t key_width = key_weight->shape[1];
  const uint64_t value_input_length = value_weight->shape[0];
  const uint64_t value_width = value_weight->shape[1];
  const uint64_t head_dim_u64 = static_cast<uint64_t>(head_dim);
  if (embedding_length == 0 || vocab_size == 0 || query_input_length == 0 ||
      query_width == 0 || key_input_length == 0 || key_width == 0 ||
      value_input_length == 0 || value_width == 0) {
    throw std::runtime_error("Multi-head attention tensors have an invalid shape");
  }
  if (norm->element_count != embedding_length) {
    throw std::runtime_error("Norm tensor length does not match embedding");
  }
  if (query_input_length != embedding_length ||
      key_input_length != embedding_length ||
      value_input_length != embedding_length) {
    throw std::runtime_error(
        "Multi-head attention projection input width does not match embedding");
  }
  if (key_width != value_width) {
    throw std::runtime_error(
        "Multi-head attention key and value widths must match");
  }
  if (query_width % head_dim_u64 != 0 || key_width % head_dim_u64 != 0) {
    throw std::runtime_error(
        "Multi-head attention projection widths must be divisible by headDim");
  }
  const uint64_t query_head_count = query_width / head_dim_u64;
  const uint64_t kv_head_count = key_width / head_dim_u64;
  if (query_head_count == 0 || kv_head_count == 0 ||
      query_head_count % kv_head_count != 0) {
    throw std::runtime_error(
        "Multi-head attention query heads must be a multiple of KV heads");
  }
  const uint64_t group_size = query_head_count / kv_head_count;
  if (static_cast<uint64_t>(query_token_id) >= vocab_size) {
    throw std::runtime_error("Query token id is outside token_embd.weight vocab");
  }
  for (const int token_id : token_ids) {
    if (static_cast<uint64_t>(token_id) >= vocab_size) {
      throw std::runtime_error("Token id is outside token_embd.weight vocab");
    }
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  FileDescriptor fd(path);
  const std::vector<float> norm_weights = DecodeTensorVector(fd.get(), *norm);
  const uint64_t sequence_length = static_cast<uint64_t>(token_ids.size());
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  key_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, key_width),
                                "Multi-head attention key cache too large"));
  value_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, value_width),
                                  "Multi-head attention value cache too large"));

  for (size_t index_in_sequence = 0; index_in_sequence < token_ids.size();
       ++index_in_sequence) {
    const int token_id = token_ids[index_in_sequence];
    const std::vector<float> input =
        DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                        embedding_length);
    const RmsNormOutput normalized = ApplyRmsNorm(input, norm_weights, epsilon);
    std::vector<float> key =
        ProjectVectorRows(fd.get(), *key_weight, normalized.values);
    std::vector<float> value =
        ProjectVectorRows(fd.get(), *value_weight, normalized.values);
    const int64_t absolute_position =
        start_position + static_cast<int64_t>(index_in_sequence);
    ApplyRope(key, static_cast<uint64_t>(absolute_position), head_dim_u64,
              resolved_rope_theta);
    key_cache.insert(key_cache.end(), key.begin(), key.end());
    value_cache.insert(value_cache.end(), value.begin(), value.end());
  }

  const std::vector<float> query_input =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(query_token_id),
                      embedding_length);
  const RmsNormOutput query_normalized =
      ApplyRmsNorm(query_input, norm_weights, epsilon);
  std::vector<float> query_full =
      ProjectVectorRows(fd.get(), *query_weight, query_normalized.values);
  ApplyRope(query_full, static_cast<uint64_t>(query_position), head_dim_u64,
            resolved_rope_theta);

  const size_t head_dim_size =
      CheckedSize(head_dim_u64, "Multi-head attention headDim too large");
  std::vector<float> output(CheckedSize(query_width,
                                        "Multi-head attention output too large"),
                            0.0f);
  std::vector<double> first_head_scores;
  std::vector<double> first_head_probabilities;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_u64));

  for (uint64_t query_head = 0; query_head < query_head_count; ++query_head) {
    const uint64_t kv_head = query_head / group_size;
    const size_t query_offset =
        CheckedSize(CheckedMultiply(query_head, head_dim_u64),
                    "Multi-head attention query offset too large");
    const size_t kv_offset =
        CheckedSize(CheckedMultiply(kv_head, head_dim_u64),
                    "Multi-head attention KV offset too large");

    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(resolved_attention_length));
    double max_score = -std::numeric_limits<double>::infinity();
    for (int token_index = 0; token_index < resolved_attention_length;
         ++token_index) {
      const size_t key_base =
          CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                      key_width),
                      "Multi-head attention key offset too large") +
          kv_offset;
      double dot = 0.0;
      for (size_t dim = 0; dim < head_dim_size; ++dim) {
        dot += static_cast<double>(query_full[query_offset + dim]) *
               static_cast<double>(key_cache[key_base + dim]);
      }
      const double score = dot * scale;
      scores.push_back(score);
      max_score = std::max(max_score, score);
    }

    std::vector<double> probabilities;
    probabilities.reserve(scores.size());
    double sum_exp = 0.0;
    for (const double score : scores) {
      const double value = std::exp(score - max_score);
      probabilities.push_back(value);
      sum_exp += value;
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
      throw std::runtime_error(
          "Multi-head attention softmax produced an invalid sum");
    }
    for (double& probability : probabilities) {
      probability /= sum_exp;
    }

    for (int token_index = 0; token_index < resolved_attention_length;
         ++token_index) {
      const size_t value_base =
          CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                      value_width),
                      "Multi-head attention value offset too large") +
          kv_offset;
      const double probability =
          probabilities[static_cast<size_t>(token_index)];
      for (size_t dim = 0; dim < head_dim_size; ++dim) {
        output[query_offset + dim] = static_cast<float>(
            static_cast<double>(output[query_offset + dim]) +
            probability * static_cast<double>(value_cache[value_base + dim]));
      }
    }

    if (query_head == 0) {
      first_head_scores = std::move(scores);
      first_head_probabilities = std::move(probabilities);
    }
  }

  const VectorStats output_stats = ComputeStats(output);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_output_values =
      std::min<size_t>(static_cast<size_t>(preview_count), output.size());
  const size_t returned_query_values =
      std::min<size_t>(static_cast<size_t>(preview_count), query_full.size());
  const size_t returned_score_values = std::min<size_t>(
      static_cast<size_t>(preview_count), first_head_scores.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "queryTokenId",
                        static_cast<uint64_t>(query_token_id), first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "normTensorName", norm->name, first);
  AppendJsonField(out, "normTensorType", TensorTypeName(norm->type), first);
  AppendJsonField(out, "queryWeightTensorName", query_weight->name, first);
  AppendJsonField(out, "queryWeightTensorType",
                  TensorTypeName(query_weight->type), first);
  AppendJsonField(out, "keyWeightTensorName", key_weight->name, first);
  AppendJsonField(out, "keyWeightTensorType", TensorTypeName(key_weight->type),
                  first);
  AppendJsonField(out, "valueWeightTensorName", value_weight->name, first);
  AppendJsonField(out, "valueWeightTensorType",
                  TensorTypeName(value_weight->type), first);
  AppendJsonNumberField(out, "sequenceLength", sequence_length, first);
  AppendJsonNumberField(out, "attentionLength",
                        static_cast<uint64_t>(resolved_attention_length),
                        first);
  AppendJsonNumberField(out, "embeddingLength", embedding_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "queryWidth", query_width, first);
  AppendJsonNumberField(out, "keyWidth", key_width, first);
  AppendJsonNumberField(out, "valueWidth", value_width, first);
  AppendJsonNumberField(out, "headDim", head_dim_u64, first);
  AppendJsonNumberField(out, "queryHeadCount", query_head_count, first);
  AppendJsonNumberField(out, "kvHeadCount", kv_head_count, first);
  AppendJsonNumberField(out, "groupSize", group_size, first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "queryPosition",
                        static_cast<uint64_t>(query_position), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(read_index), first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedOutputValueCount",
                        static_cast<uint64_t>(returned_output_values), first);
  AppendJsonNumberField(out, "returnedScoreValueCount",
                        static_cast<uint64_t>(returned_score_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "queryMeanSquare", query_normalized.mean_square,
                       first);
  AppendJsonFloatField(out, "queryInvRms", query_normalized.inv_rms, first);
  AppendJsonFloatField(out, "ropeTheta", resolved_rope_theta, first);
  AppendJsonFloatField(out, "scale", scale, first);
  AppendJsonFloatField(out, "outputMin", output_stats.min, first);
  AppendJsonFloatField(out, "outputMax", output_stats.max, first);
  AppendJsonFloatField(out, "outputMean", output_stats.mean, first);
  AppendJsonFloatField(out, "outputL2Norm", output_stats.l2_norm, first);
  AppendJsonFloatField(out, "outputChecksum", output_stats.checksum, first);

  out << "," << Quote("tokenIds") << ":[";
  for (size_t i = 0; i < token_ids.size(); ++i) {
    if (i != 0) out << ",";
    out << token_ids[i];
  }
  out << "]";

  out << "," << Quote("qHeadToKvHead") << ":[";
  for (uint64_t query_head = 0; query_head < query_head_count; ++query_head) {
    if (query_head != 0) out << ",";
    out << (query_head / group_size);
  }
  out << "]";

  out << "," << Quote("queryValues") << ":[";
  for (size_t i = 0; i < returned_query_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(query_full[i])) {
      out << std::setprecision(9) << query_full[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("firstHeadScores") << ":[";
  for (size_t i = 0; i < returned_score_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(first_head_scores[i])) {
      out << std::setprecision(9) << first_head_scores[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("firstHeadProbabilities") << ":[";
  for (size_t i = 0; i < returned_score_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(first_head_probabilities[i])) {
      out << std::setprecision(9) << first_head_probabilities[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("outputValues") << ":[";
  for (size_t i = 0; i < returned_output_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(output[i])) {
      out << std::setprecision(9) << output[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

struct DecoderLayerRun {
  std::vector<std::vector<float>> hidden_states;
  std::vector<float> query_values;
  std::vector<float> attention_values;
  std::vector<float> attention_projected_values;
  std::vector<float> post_attention_values;
  std::vector<float> gate_values;
  std::vector<float> up_values;
  std::vector<float> ffn_hidden_values;
  std::vector<float> ffn_output_values;
  std::vector<float> output_values;
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  std::vector<double> first_head_scores;
  std::vector<double> first_head_probabilities;
  std::vector<int> q_head_to_kv_head;
  uint64_t query_width = 0;
  uint64_t key_width = 0;
  uint64_t value_width = 0;
  uint64_t ffn_width = 0;
  uint64_t query_head_count = 0;
  uint64_t kv_head_count = 0;
  uint64_t group_size = 0;
  uint64_t kv_cache_element_count = 0;
  double attn_mean_square = 0.0;
  double attn_inv_rms = 0.0;
  double ffn_mean_square = 0.0;
  double ffn_inv_rms = 0.0;
};

struct LayerKvCache {
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  std::vector<int> q_head_to_kv_head;
  uint64_t token_count = 0;
  uint64_t query_width = 0;
  uint64_t key_width = 0;
  uint64_t value_width = 0;
  uint64_t ffn_width = 0;
  uint64_t query_head_count = 0;
  uint64_t kv_head_count = 0;
  uint64_t group_size = 0;
};

struct TransformerStackRun {
  GgufIndex index;
  std::string embedding_tensor_name;
  uint32_t embedding_tensor_type = 0;
  uint64_t block_count = 0;
  uint64_t hidden_size = 0;
  uint64_t vocab_size = 0;
  uint64_t kv_cache_element_count = 0;
  int read_index = 0;
  int resolved_layer_count = 0;
  double resolved_rope_theta = 0.0;
  DecoderLayerRun last_layer;
  std::vector<LayerKvCache> layer_caches;
  std::vector<double> layer_output_checksums;
  std::vector<float> selected_output;
};

struct NextTokenProjection {
  RmsNormOutput normalized;
  std::vector<float> logits;
  std::vector<int> top_token_ids;
  std::vector<float> top_logits;
  int next_token_id = 0;
  float next_token_logit = 0.0f;
};

struct GgufSession {
  std::string path;
  GgufIndex index;
  std::string attn_norm_tensor_suffix;
  std::string query_weight_tensor_suffix;
  std::string key_weight_tensor_suffix;
  std::string value_weight_tensor_suffix;
  std::string output_weight_tensor_suffix;
  std::string ffn_norm_tensor_suffix;
  std::string gate_weight_tensor_suffix;
  std::string up_weight_tensor_suffix;
  std::string down_weight_tensor_suffix;
  std::string final_norm_tensor_name;
  std::string lm_head_tensor_name;
  float epsilon = 0.0f;
  int64_t start_position = 0;
  int64_t next_position = 0;
  int resolved_layer_count = 0;
  int head_dim = 0;
  double resolved_rope_theta = 0.0;
  uint64_t block_count = 0;
  uint64_t hidden_size = 0;
  uint64_t vocab_size = 0;
  uint64_t initial_kv_cache_element_count = 0;
  std::vector<int> prompt_token_ids;
  std::vector<int> all_token_ids;
  std::vector<int> generated_token_ids;
  std::vector<float> generated_token_logits;
  std::vector<float> current_hidden;
  std::vector<LayerKvCache> layer_caches;
  std::vector<int> q_head_to_kv_head;
  std::vector<double> layer_output_checksums;
  uint64_t query_head_count = 0;
  uint64_t kv_head_count = 0;
  uint64_t group_size = 0;
};

std::mutex g_sessions_mutex;
std::map<uint64_t, GgufSession> g_sessions;
uint64_t g_next_session_id = 1;

const TensorInfo* RequireKnownTensor(const GgufIndex& index,
                                     const std::string& name,
                                     const std::string& role) {
  const TensorInfo* tensor = FindTensor(index, name);
  if (tensor == nullptr) {
    throw std::runtime_error("GGUF " + role + " tensor was not found: " +
                             name);
  }
  if (!tensor->has_known_byte_size) {
    throw std::runtime_error("GGUF " + role +
                             " tensor uses an unsupported GGML type: " + name);
  }
  return tensor;
}

std::string LayerTensorName(int layer_index, const std::string& suffix) {
  if (suffix.empty()) {
    throw std::runtime_error("Transformer stack tensor suffix is empty");
  }
  return "blk." + std::to_string(layer_index) + "." + suffix;
}

DecoderLayerRun RunDecoderLayer(
    int fd, const GgufIndex& index,
    const std::vector<std::vector<float>>& input_states, int layer_index,
    int read_index, const std::string& attn_norm_tensor_suffix,
    const std::string& query_weight_tensor_suffix,
    const std::string& key_weight_tensor_suffix,
    const std::string& value_weight_tensor_suffix,
    const std::string& output_weight_tensor_suffix,
    const std::string& ffn_norm_tensor_suffix,
    const std::string& gate_weight_tensor_suffix,
    const std::string& up_weight_tensor_suffix,
    const std::string& down_weight_tensor_suffix, float epsilon,
    int64_t start_position, uint64_t head_dim, double rope_theta) {
  if (input_states.empty()) {
    throw std::runtime_error("Transformer stack layer input is empty");
  }
  const uint64_t sequence_length = static_cast<uint64_t>(input_states.size());
  const uint64_t hidden_size = static_cast<uint64_t>(input_states.front().size());
  if (hidden_size == 0) {
    throw std::runtime_error("Transformer stack hidden size is zero");
  }
  for (const std::vector<float>& state : input_states) {
    if (state.size() != input_states.front().size()) {
      throw std::runtime_error(
          "Transformer stack hidden state sizes do not match");
    }
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >= sequence_length) {
    throw std::runtime_error("Transformer stack readPosition is outside input");
  }
  if (head_dim == 0) {
    throw std::runtime_error("Transformer stack headDim must be positive");
  }

  const TensorInfo* attn_norm = RequireKnownTensor(
      index, LayerTensorName(layer_index, attn_norm_tensor_suffix),
      "stack attention norm");
  const TensorInfo* query_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, query_weight_tensor_suffix),
      "stack query weight");
  const TensorInfo* key_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, key_weight_tensor_suffix),
      "stack key weight");
  const TensorInfo* value_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, value_weight_tensor_suffix),
      "stack value weight");
  const TensorInfo* output_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, output_weight_tensor_suffix),
      "stack attention output weight");
  const TensorInfo* ffn_norm = RequireKnownTensor(
      index, LayerTensorName(layer_index, ffn_norm_tensor_suffix),
      "stack FFN norm");
  const TensorInfo* gate_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, gate_weight_tensor_suffix),
      "stack FFN gate weight");
  const TensorInfo* up_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, up_weight_tensor_suffix),
      "stack FFN up weight");
  const TensorInfo* down_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, down_weight_tensor_suffix),
      "stack FFN down weight");

  if (query_weight->shape.size() < 2 || key_weight->shape.size() < 2 ||
      value_weight->shape.size() < 2 || output_weight->shape.size() < 2 ||
      gate_weight->shape.size() < 2 || up_weight->shape.size() < 2 ||
      down_weight->shape.size() < 2) {
    throw std::runtime_error(
        "Transformer stack projection weight tensors must be 2D");
  }

  const uint64_t query_input_length = query_weight->shape[0];
  const uint64_t query_width = query_weight->shape[1];
  const uint64_t key_input_length = key_weight->shape[0];
  const uint64_t key_width = key_weight->shape[1];
  const uint64_t value_input_length = value_weight->shape[0];
  const uint64_t value_width = value_weight->shape[1];
  const uint64_t output_input_length = output_weight->shape[0];
  const uint64_t output_width = output_weight->shape[1];
  const uint64_t gate_input_length = gate_weight->shape[0];
  const uint64_t gate_width = gate_weight->shape[1];
  const uint64_t up_input_length = up_weight->shape[0];
  const uint64_t up_width = up_weight->shape[1];
  const uint64_t down_input_length = down_weight->shape[0];
  const uint64_t down_width = down_weight->shape[1];
  if (attn_norm->element_count != hidden_size ||
      ffn_norm->element_count != hidden_size) {
    throw std::runtime_error(
        "Transformer stack norm tensor length does not match hidden size");
  }
  if (query_input_length != hidden_size || key_input_length != hidden_size ||
      value_input_length != hidden_size || gate_input_length != hidden_size ||
      up_input_length != hidden_size) {
    throw std::runtime_error(
        "Transformer stack projection input width does not match hidden size");
  }
  if (output_input_length != query_width || output_width != hidden_size) {
    throw std::runtime_error(
        "Transformer stack attention output projection shape is incompatible");
  }
  if (key_width != value_width) {
    throw std::runtime_error("Transformer stack key/value widths must match");
  }
  if (query_width % head_dim != 0 || key_width % head_dim != 0) {
    throw std::runtime_error(
        "Transformer stack attention widths must be divisible by headDim");
  }
  const uint64_t query_head_count = query_width / head_dim;
  const uint64_t kv_head_count = key_width / head_dim;
  if (query_head_count == 0 || kv_head_count == 0 ||
      query_head_count % kv_head_count != 0) {
    throw std::runtime_error(
        "Transformer stack query heads must be a multiple of KV heads");
  }
  const uint64_t group_size = query_head_count / kv_head_count;
  if (gate_width != up_width || down_input_length != gate_width ||
      down_width != hidden_size) {
    throw std::runtime_error(
        "Transformer stack FFN gate/up/down shapes are incompatible");
  }

  const std::vector<float> attn_norm_weights =
      DecodeTensorVector(fd, *attn_norm);
  const std::vector<float> ffn_norm_weights =
      DecodeTensorVector(fd, *ffn_norm);
  std::vector<std::vector<float>> attn_inputs;
  attn_inputs.reserve(input_states.size());
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  key_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, key_width),
                                "Transformer stack key cache too large"));
  value_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, value_width),
                                  "Transformer stack value cache too large"));

  for (size_t position = 0; position < input_states.size(); ++position) {
    const RmsNormOutput normalized =
        ApplyRmsNorm(input_states[position], attn_norm_weights, epsilon);
    attn_inputs.push_back(normalized.values);
    std::vector<float> key =
        ProjectVectorRows(fd, *key_weight, normalized.values);
    std::vector<float> value =
        ProjectVectorRows(fd, *value_weight, normalized.values);
    const int64_t absolute_position =
        start_position + static_cast<int64_t>(position);
    ApplyRope(key, static_cast<uint64_t>(absolute_position), head_dim,
              rope_theta);
    key_cache.insert(key_cache.end(), key.begin(), key.end());
    value_cache.insert(value_cache.end(), value.begin(), value.end());
  }

  DecoderLayerRun result;
  result.hidden_states.reserve(input_states.size());
  result.query_width = query_width;
  result.key_width = key_width;
  result.value_width = value_width;
  result.ffn_width = gate_width;
  result.query_head_count = query_head_count;
  result.kv_head_count = kv_head_count;
  result.group_size = group_size;
  result.kv_cache_element_count =
      CheckedMultiply(CheckedMultiply(sequence_length, key_width), 2);
  result.q_head_to_kv_head.reserve(
      CheckedSize(query_head_count,
                  "Transformer stack query head count too large"));
  for (uint64_t query_head = 0; query_head < query_head_count; ++query_head) {
    result.q_head_to_kv_head.push_back(
        static_cast<int>(query_head / group_size));
  }

  const size_t head_dim_size =
      CheckedSize(head_dim, "Transformer stack headDim too large");
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  for (size_t position = 0; position < input_states.size(); ++position) {
    std::vector<float> query_full =
        ProjectVectorRows(fd, *query_weight, attn_inputs[position]);
    const int64_t absolute_position =
        start_position + static_cast<int64_t>(position);
    ApplyRope(query_full, static_cast<uint64_t>(absolute_position), head_dim,
              rope_theta);

    std::vector<float> attention_output(
        CheckedSize(query_width,
                    "Transformer stack attention output too large"),
        0.0f);
    std::vector<double> selected_first_scores;
    std::vector<double> selected_first_probabilities;
    const int attention_length = static_cast<int>(position + 1);

    for (uint64_t query_head = 0; query_head < query_head_count; ++query_head) {
      const uint64_t kv_head = query_head / group_size;
      const size_t query_offset =
          CheckedSize(CheckedMultiply(query_head, head_dim),
                      "Transformer stack query offset too large");
      const size_t kv_offset =
          CheckedSize(CheckedMultiply(kv_head, head_dim),
                      "Transformer stack KV offset too large");

      std::vector<double> scores;
      scores.reserve(static_cast<size_t>(attention_length));
      double max_score = -std::numeric_limits<double>::infinity();
      for (int token_index = 0; token_index < attention_length; ++token_index) {
        const size_t key_base =
            CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                        key_width),
                        "Transformer stack key offset too large") +
            kv_offset;
        double dot = 0.0;
        for (size_t dim = 0; dim < head_dim_size; ++dim) {
          dot += static_cast<double>(query_full[query_offset + dim]) *
                 static_cast<double>(key_cache[key_base + dim]);
        }
        const double score = dot * scale;
        scores.push_back(score);
        max_score = std::max(max_score, score);
      }

      std::vector<double> probabilities;
      probabilities.reserve(scores.size());
      double sum_exp = 0.0;
      for (const double score : scores) {
        const double value = std::exp(score - max_score);
        probabilities.push_back(value);
        sum_exp += value;
      }
      if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        throw std::runtime_error(
            "Transformer stack attention softmax produced an invalid sum");
      }
      for (double& probability : probabilities) {
        probability /= sum_exp;
      }

      for (int token_index = 0; token_index < attention_length; ++token_index) {
        const size_t value_base =
            CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                        value_width),
                        "Transformer stack value offset too large") +
            kv_offset;
        const double probability =
            probabilities[static_cast<size_t>(token_index)];
        for (size_t dim = 0; dim < head_dim_size; ++dim) {
          attention_output[query_offset + dim] = static_cast<float>(
              static_cast<double>(attention_output[query_offset + dim]) +
              probability *
                  static_cast<double>(value_cache[value_base + dim]));
        }
      }

      if (query_head == 0) {
        selected_first_scores = scores;
        selected_first_probabilities = probabilities;
      }
    }

    const std::vector<float> attention_projection =
        ProjectVectorRows(fd, *output_weight, attention_output);
    const std::vector<float> post_attention =
        AddVectors(input_states[position], attention_projection,
                   "Transformer stack attention residual");
    const RmsNormOutput ffn_normalized =
        ApplyRmsNorm(post_attention, ffn_norm_weights, epsilon);
    const std::vector<float> gate =
        ProjectVectorRows(fd, *gate_weight, ffn_normalized.values);
    const std::vector<float> up =
        ProjectVectorRows(fd, *up_weight, ffn_normalized.values);
    const std::vector<float> ffn_hidden = ApplySwiGlu(gate, up);
    const std::vector<float> ffn_output =
        ProjectVectorRows(fd, *down_weight, ffn_hidden);
    std::vector<float> output =
        AddVectors(post_attention, ffn_output, "Transformer stack FFN residual");

    if (static_cast<int>(position) == read_index) {
      result.query_values = query_full;
      result.attention_values = attention_output;
      result.attention_projected_values = attention_projection;
      result.post_attention_values = post_attention;
      result.gate_values = gate;
      result.up_values = up;
      result.ffn_hidden_values = ffn_hidden;
      result.ffn_output_values = ffn_output;
      result.output_values = output;
      result.first_head_scores = selected_first_scores;
      result.first_head_probabilities = selected_first_probabilities;
      const RmsNormOutput selected_attn =
          ApplyRmsNorm(input_states[position], attn_norm_weights, epsilon);
      result.attn_mean_square = selected_attn.mean_square;
      result.attn_inv_rms = selected_attn.inv_rms;
      result.ffn_mean_square = ffn_normalized.mean_square;
      result.ffn_inv_rms = ffn_normalized.inv_rms;
    }

    result.hidden_states.push_back(std::move(output));
  }

  result.key_cache = std::move(key_cache);
  result.value_cache = std::move(value_cache);
  return result;
}

TransformerStackRun RunTransformerStack(
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
    double rope_theta) {
  if (token_ids.empty()) {
    throw std::runtime_error("Transformer stack tokenIds must be non-empty");
  }
  if (token_ids.size() > 512) {
    throw std::runtime_error(
        "Transformer stack diagnostic sequence is too large");
  }
  if (start_position < 0) {
    throw std::runtime_error(
        "Transformer stack startPosition must be non-negative");
  }
  if (head_dim <= 0) {
    throw std::runtime_error("Transformer stack headDim must be positive");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error(
          "Transformer stack token ids must be non-negative");
    }
  }

  int read_index = read_position;
  if (read_index < 0) {
    read_index = static_cast<int>(token_ids.size() - 1);
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >=
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error(
        "Transformer stack readPosition is outside the sequence");
  }
  if (start_position >
      std::numeric_limits<int64_t>::max() -
          static_cast<int64_t>(token_ids.size() - 1)) {
    throw std::runtime_error("Transformer stack cache positions overflow");
  }

  GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (!embedding->has_known_byte_size) {
    throw std::runtime_error(
        "token_embd.weight uses an unsupported GGML tensor type");
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  const uint64_t hidden_size = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  if (hidden_size == 0 || vocab_size == 0) {
    throw std::runtime_error("token_embd.weight has an invalid shape");
  }
  for (const int token_id : token_ids) {
    if (static_cast<uint64_t>(token_id) >= vocab_size) {
      throw std::runtime_error("Token id is outside token_embd.weight vocab");
    }
  }

  const uint64_t block_count = ParseUnsigned(
      index.metadata, index.arch_prefix + "block_count", 0);
  int resolved_layer_count = layer_count;
  if (resolved_layer_count <= 0) {
    if (block_count == 0 ||
        block_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "Transformer stack layerCount must be provided for this GGUF");
    }
    resolved_layer_count = static_cast<int>(block_count);
  }
  if (resolved_layer_count <= 0 || resolved_layer_count > 128) {
    throw std::runtime_error(
        "Transformer stack layerCount must be between 1 and 128");
  }
  if (block_count != 0 &&
      static_cast<uint64_t>(resolved_layer_count) > block_count) {
    throw std::runtime_error(
        "Transformer stack layerCount exceeds GGUF block_count");
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  FileDescriptor fd(path);
  std::vector<std::vector<float>> hidden_states;
  hidden_states.reserve(token_ids.size());
  for (const int token_id : token_ids) {
    hidden_states.push_back(DecodeTensorRow(
        fd.get(), *embedding, static_cast<uint64_t>(token_id), hidden_size));
  }

  DecoderLayerRun last_layer;
  std::vector<LayerKvCache> layer_caches;
  layer_caches.reserve(static_cast<size_t>(resolved_layer_count));
  std::vector<double> layer_output_checksums;
  layer_output_checksums.reserve(static_cast<size_t>(resolved_layer_count));
  uint64_t kv_cache_element_count = 0;
  for (int layer_index = 0; layer_index < resolved_layer_count; ++layer_index) {
    DecoderLayerRun layer_run = RunDecoderLayer(
        fd.get(), index, hidden_states, layer_index, read_index,
        attn_norm_tensor_suffix, query_weight_tensor_suffix,
        key_weight_tensor_suffix, value_weight_tensor_suffix,
        output_weight_tensor_suffix, ffn_norm_tensor_suffix,
        gate_weight_tensor_suffix, up_weight_tensor_suffix,
        down_weight_tensor_suffix, epsilon, start_position,
        static_cast<uint64_t>(head_dim), resolved_rope_theta);
    hidden_states = layer_run.hidden_states;
    const VectorStats selected_stats =
        ComputeStats(hidden_states[static_cast<size_t>(read_index)]);
    layer_output_checksums.push_back(selected_stats.checksum);
    kv_cache_element_count =
        CheckedAdd(kv_cache_element_count, layer_run.kv_cache_element_count);

    LayerKvCache layer_cache;
    layer_cache.key_cache = layer_run.key_cache;
    layer_cache.value_cache = layer_run.value_cache;
    layer_cache.q_head_to_kv_head = layer_run.q_head_to_kv_head;
    layer_cache.token_count = static_cast<uint64_t>(token_ids.size());
    layer_cache.query_width = layer_run.query_width;
    layer_cache.key_width = layer_run.key_width;
    layer_cache.value_width = layer_run.value_width;
    layer_cache.ffn_width = layer_run.ffn_width;
    layer_cache.query_head_count = layer_run.query_head_count;
    layer_cache.kv_head_count = layer_run.kv_head_count;
    layer_cache.group_size = layer_run.group_size;
    layer_caches.push_back(std::move(layer_cache));
    last_layer = std::move(layer_run);
  }

  TransformerStackRun result;
  result.embedding_tensor_name = embedding->name;
  result.embedding_tensor_type = embedding->type;
  result.block_count = block_count;
  result.hidden_size = hidden_size;
  result.vocab_size = vocab_size;
  result.kv_cache_element_count = kv_cache_element_count;
  result.read_index = read_index;
  result.resolved_layer_count = resolved_layer_count;
  result.resolved_rope_theta = resolved_rope_theta;
  result.last_layer = std::move(last_layer);
  result.layer_caches = std::move(layer_caches);
  result.layer_output_checksums = std::move(layer_output_checksums);
  result.selected_output = hidden_states[static_cast<size_t>(read_index)];
  result.index = std::move(index);
  return result;
}

std::vector<float> RunDecoderLayerToken(
    int fd, const GgufIndex& index, const std::vector<float>& input_state,
    int layer_index, int64_t absolute_position, LayerKvCache& cache,
    const std::string& attn_norm_tensor_suffix,
    const std::string& query_weight_tensor_suffix,
    const std::string& key_weight_tensor_suffix,
    const std::string& value_weight_tensor_suffix,
    const std::string& output_weight_tensor_suffix,
    const std::string& ffn_norm_tensor_suffix,
    const std::string& gate_weight_tensor_suffix,
    const std::string& up_weight_tensor_suffix,
    const std::string& down_weight_tensor_suffix, float epsilon,
    uint64_t head_dim, double rope_theta) {
  if (input_state.empty()) {
    throw std::runtime_error("Generate token layer input is empty");
  }
  if (absolute_position < 0) {
    throw std::runtime_error(
        "Generate token absolute position must be non-negative");
  }

  const uint64_t hidden_size = static_cast<uint64_t>(input_state.size());
  const TensorInfo* attn_norm = RequireKnownTensor(
      index, LayerTensorName(layer_index, attn_norm_tensor_suffix),
      "generate attention norm");
  const TensorInfo* query_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, query_weight_tensor_suffix),
      "generate query weight");
  const TensorInfo* key_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, key_weight_tensor_suffix),
      "generate key weight");
  const TensorInfo* value_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, value_weight_tensor_suffix),
      "generate value weight");
  const TensorInfo* output_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, output_weight_tensor_suffix),
      "generate attention output weight");
  const TensorInfo* ffn_norm = RequireKnownTensor(
      index, LayerTensorName(layer_index, ffn_norm_tensor_suffix),
      "generate FFN norm");
  const TensorInfo* gate_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, gate_weight_tensor_suffix),
      "generate FFN gate weight");
  const TensorInfo* up_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, up_weight_tensor_suffix),
      "generate FFN up weight");
  const TensorInfo* down_weight = RequireKnownTensor(
      index, LayerTensorName(layer_index, down_weight_tensor_suffix),
      "generate FFN down weight");

  if (query_weight->shape.size() < 2 || key_weight->shape.size() < 2 ||
      value_weight->shape.size() < 2 || output_weight->shape.size() < 2 ||
      gate_weight->shape.size() < 2 || up_weight->shape.size() < 2 ||
      down_weight->shape.size() < 2) {
    throw std::runtime_error(
        "Generate token projection weight tensors must be 2D");
  }

  const uint64_t query_width = query_weight->shape[1];
  const uint64_t key_width = key_weight->shape[1];
  const uint64_t value_width = value_weight->shape[1];
  const uint64_t gate_width = gate_weight->shape[1];
  if (attn_norm->element_count != hidden_size ||
      ffn_norm->element_count != hidden_size ||
      query_weight->shape[0] != hidden_size ||
      key_weight->shape[0] != hidden_size ||
      value_weight->shape[0] != hidden_size ||
      gate_weight->shape[0] != hidden_size ||
      up_weight->shape[0] != hidden_size) {
    throw std::runtime_error(
        "Generate token layer tensor shape does not match hidden size");
  }
  if (output_weight->shape[0] != query_width ||
      output_weight->shape[1] != hidden_size ||
      up_weight->shape[1] != gate_width ||
      down_weight->shape[0] != gate_width ||
      down_weight->shape[1] != hidden_size) {
    throw std::runtime_error(
        "Generate token layer projection shapes are incompatible");
  }
  if (key_width != value_width || query_width != cache.query_width ||
      key_width != cache.key_width || value_width != cache.value_width ||
      gate_width != cache.ffn_width) {
    throw std::runtime_error(
        "Generate token cache metadata does not match layer tensors");
  }
  if (query_width % head_dim != 0 || key_width % head_dim != 0) {
    throw std::runtime_error(
        "Generate token attention widths must be divisible by headDim");
  }
  if (cache.key_cache.size() !=
          CheckedSize(CheckedMultiply(cache.token_count, cache.key_width),
                      "Generate token key cache size too large") ||
      cache.value_cache.size() !=
          CheckedSize(CheckedMultiply(cache.token_count, cache.value_width),
                      "Generate token value cache size too large")) {
    throw std::runtime_error("Generate token KV cache size is inconsistent");
  }

  const std::vector<float> attn_norm_weights =
      DecodeTensorVector(fd, *attn_norm);
  const std::vector<float> ffn_norm_weights =
      DecodeTensorVector(fd, *ffn_norm);
  const RmsNormOutput normalized =
      ApplyRmsNorm(input_state, attn_norm_weights, epsilon);
  std::vector<float> key =
      ProjectVectorRows(fd, *key_weight, normalized.values);
  std::vector<float> value =
      ProjectVectorRows(fd, *value_weight, normalized.values);
  ApplyRope(key, static_cast<uint64_t>(absolute_position), head_dim,
            rope_theta);
  cache.key_cache.insert(cache.key_cache.end(), key.begin(), key.end());
  cache.value_cache.insert(cache.value_cache.end(), value.begin(),
                           value.end());
  cache.token_count += 1;

  std::vector<float> query_full =
      ProjectVectorRows(fd, *query_weight, normalized.values);
  ApplyRope(query_full, static_cast<uint64_t>(absolute_position), head_dim,
            rope_theta);

  std::vector<float> attention_output(
      CheckedSize(query_width, "Generate token attention output too large"),
      0.0f);
  const size_t head_dim_size =
      CheckedSize(head_dim, "Generate token headDim too large");
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (uint64_t query_head = 0; query_head < cache.query_head_count;
       ++query_head) {
    const uint64_t kv_head = query_head / cache.group_size;
    const size_t query_offset =
        CheckedSize(CheckedMultiply(query_head, head_dim),
                    "Generate token query offset too large");
    const size_t kv_offset =
        CheckedSize(CheckedMultiply(kv_head, head_dim),
                    "Generate token KV offset too large");

    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(cache.token_count));
    double max_score = -std::numeric_limits<double>::infinity();
    for (uint64_t token_index = 0; token_index < cache.token_count;
         ++token_index) {
      const size_t key_base =
          CheckedSize(CheckedMultiply(token_index, cache.key_width),
                      "Generate token key offset too large") +
          kv_offset;
      double dot = 0.0;
      for (size_t dim = 0; dim < head_dim_size; ++dim) {
        dot += static_cast<double>(query_full[query_offset + dim]) *
               static_cast<double>(cache.key_cache[key_base + dim]);
      }
      const double score = dot * scale;
      scores.push_back(score);
      max_score = std::max(max_score, score);
    }

    std::vector<double> probabilities;
    probabilities.reserve(scores.size());
    double sum_exp = 0.0;
    for (const double score : scores) {
      const double value_exp = std::exp(score - max_score);
      probabilities.push_back(value_exp);
      sum_exp += value_exp;
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
      throw std::runtime_error(
          "Generate token attention softmax produced an invalid sum");
    }
    for (double& probability : probabilities) {
      probability /= sum_exp;
    }

    for (uint64_t token_index = 0; token_index < cache.token_count;
         ++token_index) {
      const size_t value_base =
          CheckedSize(CheckedMultiply(token_index, cache.value_width),
                      "Generate token value offset too large") +
          kv_offset;
      const double probability = probabilities[static_cast<size_t>(token_index)];
      for (size_t dim = 0; dim < head_dim_size; ++dim) {
        attention_output[query_offset + dim] = static_cast<float>(
            static_cast<double>(attention_output[query_offset + dim]) +
            probability *
                static_cast<double>(cache.value_cache[value_base + dim]));
      }
    }
  }

  const std::vector<float> attention_projection =
      ProjectVectorRows(fd, *output_weight, attention_output);
  const std::vector<float> post_attention =
      AddVectors(input_state, attention_projection,
                 "Generate token attention residual");
  const RmsNormOutput ffn_normalized =
      ApplyRmsNorm(post_attention, ffn_norm_weights, epsilon);
  const std::vector<float> gate =
      ProjectVectorRows(fd, *gate_weight, ffn_normalized.values);
  const std::vector<float> up =
      ProjectVectorRows(fd, *up_weight, ffn_normalized.values);
  const std::vector<float> ffn_hidden = ApplySwiGlu(gate, up);
  const std::vector<float> ffn_output =
      ProjectVectorRows(fd, *down_weight, ffn_hidden);
  return AddVectors(post_attention, ffn_output,
                    "Generate token FFN residual");
}

NextTokenProjection ProjectNextToken(
    int fd, const TensorInfo& final_norm, const TensorInfo& lm_head,
    const std::vector<float>& hidden, float epsilon, int top_k) {
  if (final_norm.element_count != hidden.size()) {
    throw std::runtime_error(
        "Generate final norm tensor length does not match hidden size");
  }
  if (lm_head.shape.size() < 2) {
    throw std::runtime_error("Generate lm_head tensor must be 2D");
  }
  if (lm_head.shape[0] != hidden.size() || lm_head.shape[1] == 0) {
    throw std::runtime_error("Generate lm_head tensor shape is incompatible");
  }
  if (lm_head.shape[1] >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Generate lm_head vocab is too large");
  }

  const std::vector<float> final_norm_weights =
      DecodeTensorVector(fd, final_norm);
  NextTokenProjection result;
  result.normalized = ApplyRmsNorm(hidden, final_norm_weights, epsilon);
  result.logits = ProjectVectorRows(fd, lm_head, result.normalized.values);
  if (result.logits.empty()) {
    throw std::runtime_error("Generate lm_head produced no logits");
  }

  result.next_token_id = 0;
  result.next_token_logit = result.logits.front();
  for (size_t i = 1; i < result.logits.size(); ++i) {
    if (result.logits[i] > result.next_token_logit) {
      result.next_token_logit = result.logits[i];
      result.next_token_id = static_cast<int>(i);
    }
  }

  int resolved_top_k = top_k;
  if (resolved_top_k < 0) resolved_top_k = 0;
  resolved_top_k = std::min<int>(resolved_top_k, 128);
  resolved_top_k =
      std::min<int>(resolved_top_k, static_cast<int>(result.logits.size()));
  result.top_token_ids.reserve(result.logits.size());
  for (size_t i = 0; i < result.logits.size(); ++i) {
    result.top_token_ids.push_back(static_cast<int>(i));
  }
  std::partial_sort(
      result.top_token_ids.begin(),
      result.top_token_ids.begin() + resolved_top_k,
      result.top_token_ids.end(), [&result](int left, int right) {
        if (result.logits[static_cast<size_t>(left)] ==
            result.logits[static_cast<size_t>(right)]) {
          return left < right;
        }
        return result.logits[static_cast<size_t>(left)] >
               result.logits[static_cast<size_t>(right)];
      });
  result.top_token_ids.resize(static_cast<size_t>(resolved_top_k));
  result.top_logits.reserve(result.top_token_ids.size());
  for (const int token_id : result.top_token_ids) {
    result.top_logits.push_back(result.logits[static_cast<size_t>(token_id)]);
  }
  return result;
}

uint64_t KvCacheElementCount(const std::vector<LayerKvCache>& layer_caches) {
  uint64_t total = 0;
  for (const LayerKvCache& cache : layer_caches) {
    total = CheckedAdd(total, static_cast<uint64_t>(cache.key_cache.size()));
    total = CheckedAdd(total, static_cast<uint64_t>(cache.value_cache.size()));
  }
  return total;
}

void ValidateSessionOutputTensors(const GgufSession& session) {
  const TensorInfo* embedding =
      RequireKnownTensor(session.index, "token_embd.weight",
                         "session token embedding");
  const TensorInfo* final_norm = RequireKnownTensor(
      session.index, session.final_norm_tensor_name, "session final norm");
  const TensorInfo* lm_head = RequireKnownTensor(
      session.index, session.lm_head_tensor_name, "session lm_head");
  if (embedding->shape.size() < 2 || embedding->shape[0] != session.hidden_size ||
      embedding->shape[1] != session.vocab_size) {
    throw std::runtime_error("Session token embedding shape is incompatible");
  }
  if (final_norm->element_count != session.hidden_size) {
    throw std::runtime_error(
        "Session final norm tensor length does not match hidden size");
  }
  if (lm_head->shape.size() < 2 || lm_head->shape[0] != session.hidden_size ||
      lm_head->shape[1] != session.vocab_size) {
    throw std::runtime_error(
        "Session lm_head vocab must match token embeddings");
  }
}

std::string CreateSession(
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
    double rope_theta, int max_values) {
  if (final_norm_tensor_name.empty() || lm_head_tensor_name.empty()) {
    throw std::runtime_error(
        "Session final norm and lm_head tensor names are required");
  }

  TransformerStackRun stack = RunTransformerStack(
      path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
      key_weight_tensor_suffix, value_weight_tensor_suffix,
      output_weight_tensor_suffix, ffn_norm_tensor_suffix,
      gate_weight_tensor_suffix, up_weight_tensor_suffix,
      down_weight_tensor_suffix, epsilon, start_position, read_position,
      layer_count, head_dim, rope_theta);
  if (stack.read_index != static_cast<int>(token_ids.size() - 1)) {
    throw std::runtime_error(
        "Session readPosition must resolve to the last prompt token");
  }

  GgufSession session;
  session.path = path;
  session.index = std::move(stack.index);
  session.attn_norm_tensor_suffix = attn_norm_tensor_suffix;
  session.query_weight_tensor_suffix = query_weight_tensor_suffix;
  session.key_weight_tensor_suffix = key_weight_tensor_suffix;
  session.value_weight_tensor_suffix = value_weight_tensor_suffix;
  session.output_weight_tensor_suffix = output_weight_tensor_suffix;
  session.ffn_norm_tensor_suffix = ffn_norm_tensor_suffix;
  session.gate_weight_tensor_suffix = gate_weight_tensor_suffix;
  session.up_weight_tensor_suffix = up_weight_tensor_suffix;
  session.down_weight_tensor_suffix = down_weight_tensor_suffix;
  session.final_norm_tensor_name = final_norm_tensor_name;
  session.lm_head_tensor_name = lm_head_tensor_name;
  session.epsilon = epsilon;
  session.start_position = start_position;
  session.next_position =
      start_position + static_cast<int64_t>(token_ids.size());
  session.resolved_layer_count = stack.resolved_layer_count;
  session.head_dim = head_dim;
  session.resolved_rope_theta = stack.resolved_rope_theta;
  session.block_count = stack.block_count;
  session.hidden_size = stack.hidden_size;
  session.vocab_size = stack.vocab_size;
  session.initial_kv_cache_element_count = stack.kv_cache_element_count;
  session.prompt_token_ids = token_ids;
  session.all_token_ids = token_ids;
  session.current_hidden = stack.selected_output;
  session.layer_caches = std::move(stack.layer_caches);
  session.q_head_to_kv_head = stack.last_layer.q_head_to_kv_head;
  session.layer_output_checksums = std::move(stack.layer_output_checksums);
  session.query_head_count = stack.last_layer.query_head_count;
  session.kv_head_count = stack.last_layer.kv_head_count;
  session.group_size = stack.last_layer.group_size;
  ValidateSessionOutputTensors(session);

  uint64_t session_id = 0;
  uint64_t session_count = 0;
  std::ostringstream out;
  {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    session_id = g_next_session_id++;
    if (session_id == 0) {
      session_id = g_next_session_id++;
    }
    auto inserted = g_sessions.emplace(session_id, std::move(session));
    session_count = static_cast<uint64_t>(g_sessions.size());

    const GgufSession& stored_session = inserted.first->second;
    const uint64_t kv_cache_element_count =
        KvCacheElementCount(stored_session.layer_caches);
    const VectorStats hidden_stats =
        ComputeStats(stored_session.current_hidden);
    int preview_count = max_values;
    if (preview_count < 0) preview_count = 0;
    preview_count = std::min<int>(preview_count, 4096);
    const size_t returned_hidden_values =
        std::min<size_t>(static_cast<size_t>(preview_count),
                         stored_session.current_hidden.size());

    out << "{";
    bool first = true;
    AppendJsonBoolField(out, "ok", true, first);
    AppendJsonNumberField(out, "sessionId", session_id, first);
    AppendJsonNumberField(out, "activeSessionCount", session_count, first);
    AppendJsonField(out, "path", stored_session.path, first);
    AppendJsonField(out, "architecture", stored_session.index.architecture,
                    first);
    AppendJsonBoolField(out, "prefilled", true, first);
    AppendJsonNumberField(out, "blockCount", stored_session.block_count,
                          first);
    AppendJsonNumberField(
        out, "layerCount",
        static_cast<uint64_t>(stored_session.resolved_layer_count), first);
    AppendJsonNumberField(out, "promptTokenCount",
                          static_cast<uint64_t>(token_ids.size()), first);
    AppendJsonNumberField(out, "totalTokenCount",
                          static_cast<uint64_t>(token_ids.size()), first);
    AppendJsonNumberField(out, "generatedTokenCount", 0, first);
    AppendJsonNumberField(out, "startPosition",
                          static_cast<uint64_t>(start_position), first);
    AppendJsonNumberField(out, "nextPosition",
                          static_cast<uint64_t>(stored_session.next_position),
                          first);
    AppendJsonNumberField(out, "hiddenSize", stored_session.hidden_size,
                          first);
    AppendJsonNumberField(out, "vocabSize", stored_session.vocab_size, first);
    AppendJsonNumberField(out, "headDim",
                          static_cast<uint64_t>(stored_session.head_dim),
                          first);
    AppendJsonNumberField(out, "queryHeadCount",
                          stored_session.query_head_count, first);
    AppendJsonNumberField(out, "kvHeadCount", stored_session.kv_head_count,
                          first);
    AppendJsonNumberField(out, "groupSize", stored_session.group_size, first);
    AppendJsonNumberField(
        out, "kvCacheLayerCount",
        static_cast<uint64_t>(stored_session.layer_caches.size()), first);
    AppendJsonNumberField(
        out, "kvCacheTokenCount",
        stored_session.layer_caches.empty()
            ? 0
            : stored_session.layer_caches.front().token_count,
        first);
    AppendJsonNumberField(out, "kvCacheElementCount", kv_cache_element_count,
                          first);
    AppendJsonNumberField(out, "kvCacheBytesFp32",
                          CheckedMultiply(kv_cache_element_count, 4), first);
    AppendJsonNumberField(
        out, "requestedValueCount",
        static_cast<uint64_t>(std::max(max_values, 0)), first);
    AppendJsonNumberField(out, "returnedHiddenValueCount",
                          static_cast<uint64_t>(returned_hidden_values),
                          first);
    AppendJsonFloatField(out, "epsilon", stored_session.epsilon, first);
    AppendJsonFloatField(out, "ropeTheta",
                         stored_session.resolved_rope_theta, first);
    AppendJsonFloatField(out, "hiddenMin", hidden_stats.min, first);
    AppendJsonFloatField(out, "hiddenMax", hidden_stats.max, first);
    AppendJsonFloatField(out, "hiddenMean", hidden_stats.mean, first);
    AppendJsonFloatField(out, "hiddenL2Norm", hidden_stats.l2_norm, first);
    AppendJsonFloatField(out, "hiddenChecksum", hidden_stats.checksum, first);
    AppendJsonIntArrayField(out, "tokenIds", stored_session.prompt_token_ids,
                            first);
    AppendJsonIntArrayField(out, "allTokenIds", stored_session.all_token_ids,
                            first);
    AppendJsonIntArrayField(out, "qHeadToKvHead",
                            stored_session.q_head_to_kv_head, first);
    AppendJsonDoubleArrayField(out, "layerOutputChecksums",
                               stored_session.layer_output_checksums,
                               stored_session.layer_output_checksums.size(),
                               first);
    AppendJsonFloatArrayField(out, "hiddenValues",
                              stored_session.current_hidden,
                              returned_hidden_values, first);
    out << "}";
  }
  return out.str();
}

std::string DecodeSession(uint64_t session_id, int top_k, int max_values) {
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  auto it = g_sessions.find(session_id);
  if (it == g_sessions.end()) {
    throw std::runtime_error("GGUF session was not found");
  }
  GgufSession& session = it->second;
  ValidateSessionOutputTensors(session);

  FileDescriptor fd(session.path);
  const TensorInfo* embedding =
      RequireKnownTensor(session.index, "token_embd.weight",
                         "session token embedding");
  const TensorInfo* final_norm = RequireKnownTensor(
      session.index, session.final_norm_tensor_name, "session final norm");
  const TensorInfo* lm_head = RequireKnownTensor(
      session.index, session.lm_head_tensor_name, "session lm_head");
  const int64_t decoded_position = session.next_position;
  const NextTokenProjection projection =
      ProjectNextToken(fd.get(), *final_norm, *lm_head, session.current_hidden,
                       session.epsilon, top_k);
  if (projection.next_token_id < 0 ||
      static_cast<uint64_t>(projection.next_token_id) >= session.vocab_size) {
    throw std::runtime_error(
        "Session decoded token is outside token embeddings");
  }

  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_hidden_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       session.current_hidden.size());
  const VectorStats hidden_stats = ComputeStats(session.current_hidden);
  std::vector<float> projected_hidden_preview;
  projected_hidden_preview.reserve(returned_hidden_values);
  projected_hidden_preview.insert(
      projected_hidden_preview.end(), session.current_hidden.begin(),
      session.current_hidden.begin() +
          static_cast<std::ptrdiff_t>(returned_hidden_values));

  std::vector<float> next_hidden =
      DecodeTensorRow(fd.get(), *embedding,
                      static_cast<uint64_t>(projection.next_token_id),
                      session.hidden_size);
  for (int layer_index = 0; layer_index < session.resolved_layer_count;
       ++layer_index) {
    next_hidden = RunDecoderLayerToken(
        fd.get(), session.index, next_hidden, layer_index, decoded_position,
        session.layer_caches[static_cast<size_t>(layer_index)],
        session.attn_norm_tensor_suffix, session.query_weight_tensor_suffix,
        session.key_weight_tensor_suffix, session.value_weight_tensor_suffix,
        session.output_weight_tensor_suffix, session.ffn_norm_tensor_suffix,
        session.gate_weight_tensor_suffix, session.up_weight_tensor_suffix,
        session.down_weight_tensor_suffix, session.epsilon,
        static_cast<uint64_t>(session.head_dim), session.resolved_rope_theta);
  }

  session.current_hidden = std::move(next_hidden);
  session.generated_token_ids.push_back(projection.next_token_id);
  session.generated_token_logits.push_back(projection.next_token_logit);
  session.all_token_ids.push_back(projection.next_token_id);
  session.next_position += 1;

  const uint64_t kv_cache_element_count =
      KvCacheElementCount(session.layer_caches);
  const VectorStats normalized_stats =
      ComputeStats(projection.normalized.values);
  const VectorStats logit_stats = ComputeStats(projection.logits);
  const size_t returned_normalized_values = std::min<size_t>(
      static_cast<size_t>(preview_count), projection.normalized.values.size());
  const size_t returned_logit_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       projection.logits.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonNumberField(out, "sessionId", session_id, first);
  AppendJsonField(out, "path", session.path, first);
  AppendJsonField(out, "architecture", session.index.architecture, first);
  AppendJsonNumberField(out, "decodedPosition",
                        static_cast<uint64_t>(decoded_position), first);
  AppendJsonNumberField(out, "nextPosition",
                        static_cast<uint64_t>(session.next_position), first);
  AppendJsonNumberField(out, "generatedTokenId",
                        static_cast<uint64_t>(projection.next_token_id),
                        first);
  AppendJsonNumberField(out, "generatedTokenCount",
                        static_cast<uint64_t>(
                            session.generated_token_ids.size()),
                        first);
  AppendJsonNumberField(out, "totalTokenCount",
                        static_cast<uint64_t>(session.all_token_ids.size()),
                        first);
  AppendJsonNumberField(out, "hiddenSize", session.hidden_size, first);
  AppendJsonNumberField(out, "vocabSize", session.vocab_size, first);
  AppendJsonNumberField(out, "logitCount", lm_head->shape[1], first);
  AppendJsonNumberField(
      out, "kvCacheTokenCount",
      session.layer_caches.empty() ? 0 : session.layer_caches.front().token_count,
      first);
  AppendJsonNumberField(out, "kvCacheElementCount", kv_cache_element_count,
                        first);
  AppendJsonNumberField(out, "kvCacheBytesFp32",
                        CheckedMultiply(kv_cache_element_count, 4), first);
  AppendJsonNumberField(out, "requestedTopK",
                        static_cast<uint64_t>(std::max(top_k, 0)), first);
  AppendJsonNumberField(out, "returnedTopK",
                        static_cast<uint64_t>(
                            projection.top_token_ids.size()),
                        first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedHiddenValueCount",
                        static_cast<uint64_t>(returned_hidden_values), first);
  AppendJsonNumberField(out, "returnedNormalizedValueCount",
                        static_cast<uint64_t>(returned_normalized_values),
                        first);
  AppendJsonNumberField(out, "returnedLogitValueCount",
                        static_cast<uint64_t>(returned_logit_values), first);
  AppendJsonFloatField(out, "generatedTokenLogit",
                       projection.next_token_logit, first);
  AppendJsonFloatField(out, "epsilon", session.epsilon, first);
  AppendJsonFloatField(out, "ropeTheta", session.resolved_rope_theta, first);
  AppendJsonFloatField(out, "finalMeanSquare",
                       projection.normalized.mean_square, first);
  AppendJsonFloatField(out, "finalInvRms", projection.normalized.inv_rms,
                       first);
  AppendJsonFloatField(out, "hiddenMin", hidden_stats.min, first);
  AppendJsonFloatField(out, "hiddenMax", hidden_stats.max, first);
  AppendJsonFloatField(out, "hiddenMean", hidden_stats.mean, first);
  AppendJsonFloatField(out, "hiddenL2Norm", hidden_stats.l2_norm, first);
  AppendJsonFloatField(out, "hiddenChecksum", hidden_stats.checksum, first);
  AppendJsonFloatField(out, "normalizedMin", normalized_stats.min, first);
  AppendJsonFloatField(out, "normalizedMax", normalized_stats.max, first);
  AppendJsonFloatField(out, "normalizedMean", normalized_stats.mean, first);
  AppendJsonFloatField(out, "normalizedL2Norm", normalized_stats.l2_norm,
                       first);
  AppendJsonFloatField(out, "normalizedChecksum", normalized_stats.checksum,
                       first);
  AppendJsonFloatField(out, "logitMin", logit_stats.min, first);
  AppendJsonFloatField(out, "logitMax", logit_stats.max, first);
  AppendJsonFloatField(out, "logitMean", logit_stats.mean, first);
  AppendJsonFloatField(out, "logitL2Norm", logit_stats.l2_norm, first);
  AppendJsonFloatField(out, "logitChecksum", logit_stats.checksum, first);
  AppendJsonIntArrayField(out, "generatedTokenIds",
                          session.generated_token_ids, first);
  AppendJsonIntArrayField(out, "allTokenIds", session.all_token_ids, first);
  AppendJsonFloatArrayField(out, "generatedTokenLogits",
                            session.generated_token_logits,
                            session.generated_token_logits.size(), first);
  AppendJsonFloatArrayField(out, "hiddenValues", projected_hidden_preview,
                            projected_hidden_preview.size(), first);
  AppendJsonFloatArrayField(out, "normalizedValues",
                            projection.normalized.values,
                            returned_normalized_values, first);
  AppendJsonFloatArrayField(out, "logitValues", projection.logits,
                            returned_logit_values, first);
  AppendJsonIntArrayField(out, "topTokenIds", projection.top_token_ids, first);
  AppendJsonFloatArrayField(out, "topLogits", projection.top_logits,
                            projection.top_logits.size(), first);
  out << "}";
  return out.str();
}

std::string CloseSession(uint64_t session_id) {
  uint64_t session_count = 0;
  {
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    auto it = g_sessions.find(session_id);
    if (it == g_sessions.end()) {
      throw std::runtime_error("GGUF session was not found");
    }
    g_sessions.erase(it);
    session_count = static_cast<uint64_t>(g_sessions.size());
  }

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonNumberField(out, "sessionId", session_id, first);
  AppendJsonBoolField(out, "closed", true, first);
  AppendJsonNumberField(out, "activeSessionCount", session_count, first);
  out << "}";
  return out.str();
}

std::string TransformerLayer(
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
    int attention_length, int head_dim, double rope_theta, int max_values) {
  if (token_ids.empty()) {
    throw std::runtime_error("Transformer layer tokenIds must be non-empty");
  }
  if (token_ids.size() > 4096) {
    throw std::runtime_error(
        "Transformer layer diagnostic sequence is too large");
  }
  if (query_token_id < 0) {
    throw std::runtime_error(
        "Transformer layer queryTokenId must be non-negative");
  }
  if (start_position < 0 || query_position < 0) {
    throw std::runtime_error(
        "Transformer layer positions must be non-negative");
  }
  if (head_dim <= 0) {
    throw std::runtime_error("Transformer layer headDim must be positive");
  }
  if (attn_norm_tensor_name.empty() || query_weight_tensor_name.empty() ||
      key_weight_tensor_name.empty() || value_weight_tensor_name.empty() ||
      output_weight_tensor_name.empty() || ffn_norm_tensor_name.empty() ||
      gate_weight_tensor_name.empty() || up_weight_tensor_name.empty() ||
      down_weight_tensor_name.empty()) {
    throw std::runtime_error("Transformer layer tensor names must be non-empty");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error(
          "Transformer layer token ids must be non-negative");
    }
  }

  int read_index = read_position;
  if (read_index < 0) {
    read_index = static_cast<int>(token_ids.size() - 1);
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >=
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error(
        "Transformer layer readPosition is outside the sequence");
  }
  int resolved_attention_length = attention_length;
  if (resolved_attention_length <= 0) {
    resolved_attention_length = read_index + 1;
  }
  if (resolved_attention_length <= 0 ||
      static_cast<uint64_t>(resolved_attention_length) >
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error(
        "Transformer layer attention length is outside the sequence");
  }
  if (start_position >
      std::numeric_limits<int64_t>::max() -
          static_cast<int64_t>(token_ids.size() - 1)) {
    throw std::runtime_error("Transformer layer cache positions overflow");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  const auto require_tensor = [&index](const std::string& name,
                                      const std::string& role) {
    const TensorInfo* tensor = FindTensor(index, name);
    if (tensor == nullptr) {
      throw std::runtime_error("GGUF " + role + " tensor was not found: " +
                               name);
    }
    if (!tensor->has_known_byte_size) {
      throw std::runtime_error("GGUF " + role +
                               " tensor uses an unsupported GGML type: " +
                               name);
    }
    return tensor;
  };

  const TensorInfo* attn_norm =
      require_tensor(attn_norm_tensor_name, "attention norm");
  const TensorInfo* query_weight =
      require_tensor(query_weight_tensor_name, "query weight");
  const TensorInfo* key_weight =
      require_tensor(key_weight_tensor_name, "key weight");
  const TensorInfo* value_weight =
      require_tensor(value_weight_tensor_name, "value weight");
  const TensorInfo* output_weight =
      require_tensor(output_weight_tensor_name, "attention output weight");
  const TensorInfo* ffn_norm =
      require_tensor(ffn_norm_tensor_name, "FFN norm");
  const TensorInfo* gate_weight =
      require_tensor(gate_weight_tensor_name, "FFN gate weight");
  const TensorInfo* up_weight =
      require_tensor(up_weight_tensor_name, "FFN up weight");
  const TensorInfo* down_weight =
      require_tensor(down_weight_tensor_name, "FFN down weight");

  if (!embedding->has_known_byte_size) {
    throw std::runtime_error(
        "token_embd.weight uses an unsupported GGML tensor type");
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  if (query_weight->shape.size() < 2 || key_weight->shape.size() < 2 ||
      value_weight->shape.size() < 2 || output_weight->shape.size() < 2 ||
      gate_weight->shape.size() < 2 || up_weight->shape.size() < 2 ||
      down_weight->shape.size() < 2) {
    throw std::runtime_error(
        "Transformer layer projection weight tensors must be 2D");
  }

  const uint64_t embedding_length = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  const uint64_t query_input_length = query_weight->shape[0];
  const uint64_t query_width = query_weight->shape[1];
  const uint64_t key_input_length = key_weight->shape[0];
  const uint64_t key_width = key_weight->shape[1];
  const uint64_t value_input_length = value_weight->shape[0];
  const uint64_t value_width = value_weight->shape[1];
  const uint64_t output_input_length = output_weight->shape[0];
  const uint64_t output_width = output_weight->shape[1];
  const uint64_t gate_input_length = gate_weight->shape[0];
  const uint64_t gate_width = gate_weight->shape[1];
  const uint64_t up_input_length = up_weight->shape[0];
  const uint64_t up_width = up_weight->shape[1];
  const uint64_t down_input_length = down_weight->shape[0];
  const uint64_t down_width = down_weight->shape[1];
  const uint64_t head_dim_u64 = static_cast<uint64_t>(head_dim);
  if (embedding_length == 0 || vocab_size == 0 || query_width == 0 ||
      key_width == 0 || value_width == 0 || output_width == 0 ||
      gate_width == 0 || up_width == 0 || down_width == 0) {
    throw std::runtime_error("Transformer layer tensors have an invalid shape");
  }
  if (attn_norm->element_count != embedding_length ||
      ffn_norm->element_count != embedding_length) {
    throw std::runtime_error(
        "Transformer layer norm tensor length does not match embedding");
  }
  if (query_input_length != embedding_length ||
      key_input_length != embedding_length ||
      value_input_length != embedding_length ||
      gate_input_length != embedding_length ||
      up_input_length != embedding_length) {
    throw std::runtime_error(
        "Transformer layer projection input width does not match embedding");
  }
  if (output_input_length != query_width || output_width != embedding_length) {
    throw std::runtime_error(
        "Attention output projection must map attention width to embedding");
  }
  if (key_width != value_width) {
    throw std::runtime_error(
        "Transformer layer key and value widths must match");
  }
  if (query_width % head_dim_u64 != 0 || key_width % head_dim_u64 != 0) {
    throw std::runtime_error(
        "Transformer layer attention widths must be divisible by headDim");
  }
  const uint64_t query_head_count = query_width / head_dim_u64;
  const uint64_t kv_head_count = key_width / head_dim_u64;
  if (query_head_count == 0 || kv_head_count == 0 ||
      query_head_count % kv_head_count != 0) {
    throw std::runtime_error(
        "Transformer layer query heads must be a multiple of KV heads");
  }
  const uint64_t group_size = query_head_count / kv_head_count;
  if (gate_width != up_width || down_input_length != gate_width ||
      down_width != embedding_length) {
    throw std::runtime_error(
        "Transformer layer FFN gate/up/down shapes are incompatible");
  }
  if (static_cast<uint64_t>(query_token_id) >= vocab_size) {
    throw std::runtime_error("Query token id is outside token_embd.weight vocab");
  }
  for (const int token_id : token_ids) {
    if (static_cast<uint64_t>(token_id) >= vocab_size) {
      throw std::runtime_error("Token id is outside token_embd.weight vocab");
    }
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  FileDescriptor fd(path);
  const std::vector<float> attn_norm_weights =
      DecodeTensorVector(fd.get(), *attn_norm);
  const std::vector<float> ffn_norm_weights =
      DecodeTensorVector(fd.get(), *ffn_norm);
  const uint64_t sequence_length = static_cast<uint64_t>(token_ids.size());
  std::vector<float> key_cache;
  std::vector<float> value_cache;
  key_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, key_width),
                                "Transformer layer key cache too large"));
  value_cache.reserve(CheckedSize(CheckedMultiply(sequence_length, value_width),
                                  "Transformer layer value cache too large"));

  for (size_t index_in_sequence = 0; index_in_sequence < token_ids.size();
       ++index_in_sequence) {
    const int token_id = token_ids[index_in_sequence];
    const std::vector<float> input =
        DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(token_id),
                        embedding_length);
    const RmsNormOutput normalized =
        ApplyRmsNorm(input, attn_norm_weights, epsilon);
    std::vector<float> key =
        ProjectVectorRows(fd.get(), *key_weight, normalized.values);
    std::vector<float> value =
        ProjectVectorRows(fd.get(), *value_weight, normalized.values);
    const int64_t absolute_position =
        start_position + static_cast<int64_t>(index_in_sequence);
    ApplyRope(key, static_cast<uint64_t>(absolute_position), head_dim_u64,
              resolved_rope_theta);
    key_cache.insert(key_cache.end(), key.begin(), key.end());
    value_cache.insert(value_cache.end(), value.begin(), value.end());
  }

  const std::vector<float> residual_input =
      DecodeTensorRow(fd.get(), *embedding, static_cast<uint64_t>(query_token_id),
                      embedding_length);
  const RmsNormOutput attn_normalized =
      ApplyRmsNorm(residual_input, attn_norm_weights, epsilon);
  std::vector<float> query_full =
      ProjectVectorRows(fd.get(), *query_weight, attn_normalized.values);
  ApplyRope(query_full, static_cast<uint64_t>(query_position), head_dim_u64,
            resolved_rope_theta);

  const size_t head_dim_size =
      CheckedSize(head_dim_u64, "Transformer layer headDim too large");
  std::vector<float> attention_output(
      CheckedSize(query_width, "Transformer layer attention output too large"),
      0.0f);
  std::vector<double> first_head_scores;
  std::vector<double> first_head_probabilities;
  std::vector<int> q_head_to_kv_head;
  q_head_to_kv_head.reserve(
      CheckedSize(query_head_count,
                  "Transformer layer query head count too large"));
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_u64));

  for (uint64_t query_head = 0; query_head < query_head_count; ++query_head) {
    const uint64_t kv_head = query_head / group_size;
    q_head_to_kv_head.push_back(static_cast<int>(kv_head));
    const size_t query_offset =
        CheckedSize(CheckedMultiply(query_head, head_dim_u64),
                    "Transformer layer query offset too large");
    const size_t kv_offset =
        CheckedSize(CheckedMultiply(kv_head, head_dim_u64),
                    "Transformer layer KV offset too large");

    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(resolved_attention_length));
    double max_score = -std::numeric_limits<double>::infinity();
    for (int token_index = 0; token_index < resolved_attention_length;
         ++token_index) {
      const size_t key_base =
          CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                      key_width),
                      "Transformer layer key offset too large") +
          kv_offset;
      double dot = 0.0;
      for (size_t dim = 0; dim < head_dim_size; ++dim) {
        dot += static_cast<double>(query_full[query_offset + dim]) *
               static_cast<double>(key_cache[key_base + dim]);
      }
      const double score = dot * scale;
      scores.push_back(score);
      max_score = std::max(max_score, score);
    }

    std::vector<double> probabilities;
    probabilities.reserve(scores.size());
    double sum_exp = 0.0;
    for (const double score : scores) {
      const double value = std::exp(score - max_score);
      probabilities.push_back(value);
      sum_exp += value;
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
      throw std::runtime_error(
          "Transformer layer attention softmax produced an invalid sum");
    }
    for (double& probability : probabilities) {
      probability /= sum_exp;
    }

    for (int token_index = 0; token_index < resolved_attention_length;
         ++token_index) {
      const size_t value_base =
          CheckedSize(CheckedMultiply(static_cast<uint64_t>(token_index),
                                      value_width),
                      "Transformer layer value offset too large") +
          kv_offset;
      const double probability =
          probabilities[static_cast<size_t>(token_index)];
      for (size_t dim = 0; dim < head_dim_size; ++dim) {
        attention_output[query_offset + dim] = static_cast<float>(
            static_cast<double>(attention_output[query_offset + dim]) +
            probability * static_cast<double>(value_cache[value_base + dim]));
      }
    }

    if (query_head == 0) {
      first_head_scores = std::move(scores);
      first_head_probabilities = std::move(probabilities);
    }
  }

  const std::vector<float> attention_projection =
      ProjectVectorRows(fd.get(), *output_weight, attention_output);
  const std::vector<float> post_attention =
      AddVectors(residual_input, attention_projection,
                 "Transformer layer attention residual");
  const RmsNormOutput ffn_normalized =
      ApplyRmsNorm(post_attention, ffn_norm_weights, epsilon);
  const std::vector<float> gate =
      ProjectVectorRows(fd.get(), *gate_weight, ffn_normalized.values);
  const std::vector<float> up =
      ProjectVectorRows(fd.get(), *up_weight, ffn_normalized.values);
  const std::vector<float> ffn_hidden = ApplySwiGlu(gate, up);
  const std::vector<float> ffn_output =
      ProjectVectorRows(fd.get(), *down_weight, ffn_hidden);
  const std::vector<float> layer_output =
      AddVectors(post_attention, ffn_output, "Transformer layer FFN residual");

  const VectorStats attention_stats = ComputeStats(attention_output);
  const VectorStats attention_projection_stats =
      ComputeStats(attention_projection);
  const VectorStats post_attention_stats = ComputeStats(post_attention);
  const VectorStats gate_stats = ComputeStats(gate);
  const VectorStats up_stats = ComputeStats(up);
  const VectorStats ffn_hidden_stats = ComputeStats(ffn_hidden);
  const VectorStats ffn_output_stats = ComputeStats(ffn_output);
  const VectorStats layer_output_stats = ComputeStats(layer_output);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_output_values =
      std::min<size_t>(static_cast<size_t>(preview_count), layer_output.size());
  const size_t returned_attention_values = std::min<size_t>(
      static_cast<size_t>(preview_count), attention_output.size());
  const size_t returned_ffn_values =
      std::min<size_t>(static_cast<size_t>(preview_count), ffn_hidden.size());
  const size_t returned_score_values = std::min<size_t>(
      static_cast<size_t>(preview_count), first_head_scores.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "queryTokenId",
                        static_cast<uint64_t>(query_token_id), first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "attnNormTensorName", attn_norm->name, first);
  AppendJsonField(out, "attnNormTensorType", TensorTypeName(attn_norm->type),
                  first);
  AppendJsonField(out, "queryWeightTensorName", query_weight->name, first);
  AppendJsonField(out, "queryWeightTensorType",
                  TensorTypeName(query_weight->type), first);
  AppendJsonField(out, "keyWeightTensorName", key_weight->name, first);
  AppendJsonField(out, "keyWeightTensorType", TensorTypeName(key_weight->type),
                  first);
  AppendJsonField(out, "valueWeightTensorName", value_weight->name, first);
  AppendJsonField(out, "valueWeightTensorType",
                  TensorTypeName(value_weight->type), first);
  AppendJsonField(out, "outputWeightTensorName", output_weight->name, first);
  AppendJsonField(out, "outputWeightTensorType",
                  TensorTypeName(output_weight->type), first);
  AppendJsonField(out, "ffnNormTensorName", ffn_norm->name, first);
  AppendJsonField(out, "ffnNormTensorType", TensorTypeName(ffn_norm->type),
                  first);
  AppendJsonField(out, "gateWeightTensorName", gate_weight->name, first);
  AppendJsonField(out, "gateWeightTensorType",
                  TensorTypeName(gate_weight->type), first);
  AppendJsonField(out, "upWeightTensorName", up_weight->name, first);
  AppendJsonField(out, "upWeightTensorType", TensorTypeName(up_weight->type),
                  first);
  AppendJsonField(out, "downWeightTensorName", down_weight->name, first);
  AppendJsonField(out, "downWeightTensorType",
                  TensorTypeName(down_weight->type), first);
  AppendJsonNumberField(out, "sequenceLength", sequence_length, first);
  AppendJsonNumberField(out, "attentionLength",
                        static_cast<uint64_t>(resolved_attention_length),
                        first);
  AppendJsonNumberField(out, "embeddingLength", embedding_length, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "queryWidth", query_width, first);
  AppendJsonNumberField(out, "keyWidth", key_width, first);
  AppendJsonNumberField(out, "valueWidth", value_width, first);
  AppendJsonNumberField(out, "attentionOutputWidth", query_width, first);
  AppendJsonNumberField(out, "hiddenSize", embedding_length, first);
  AppendJsonNumberField(out, "ffnWidth", gate_width, first);
  AppendJsonNumberField(out, "headDim", head_dim_u64, first);
  AppendJsonNumberField(out, "queryHeadCount", query_head_count, first);
  AppendJsonNumberField(out, "kvHeadCount", kv_head_count, first);
  AppendJsonNumberField(out, "groupSize", group_size, first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "queryPosition",
                        static_cast<uint64_t>(query_position), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(read_index), first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedOutputValueCount",
                        static_cast<uint64_t>(returned_output_values), first);
  AppendJsonNumberField(out, "returnedAttentionValueCount",
                        static_cast<uint64_t>(returned_attention_values),
                        first);
  AppendJsonNumberField(out, "returnedFfnValueCount",
                        static_cast<uint64_t>(returned_ffn_values), first);
  AppendJsonNumberField(out, "returnedScoreValueCount",
                        static_cast<uint64_t>(returned_score_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "attnMeanSquare", attn_normalized.mean_square,
                       first);
  AppendJsonFloatField(out, "attnInvRms", attn_normalized.inv_rms, first);
  AppendJsonFloatField(out, "ffnMeanSquare", ffn_normalized.mean_square, first);
  AppendJsonFloatField(out, "ffnInvRms", ffn_normalized.inv_rms, first);
  AppendJsonFloatField(out, "ropeTheta", resolved_rope_theta, first);
  AppendJsonFloatField(out, "scale", scale, first);
  AppendJsonFloatField(out, "attentionMin", attention_stats.min, first);
  AppendJsonFloatField(out, "attentionMax", attention_stats.max, first);
  AppendJsonFloatField(out, "attentionMean", attention_stats.mean, first);
  AppendJsonFloatField(out, "attentionL2Norm", attention_stats.l2_norm, first);
  AppendJsonFloatField(out, "attentionChecksum", attention_stats.checksum,
                       first);
  AppendJsonFloatField(out, "attentionProjectionMin",
                       attention_projection_stats.min, first);
  AppendJsonFloatField(out, "attentionProjectionMax",
                       attention_projection_stats.max, first);
  AppendJsonFloatField(out, "attentionProjectionMean",
                       attention_projection_stats.mean, first);
  AppendJsonFloatField(out, "attentionProjectionL2Norm",
                       attention_projection_stats.l2_norm, first);
  AppendJsonFloatField(out, "attentionProjectionChecksum",
                       attention_projection_stats.checksum, first);
  AppendJsonFloatField(out, "postAttentionMin", post_attention_stats.min,
                       first);
  AppendJsonFloatField(out, "postAttentionMax", post_attention_stats.max,
                       first);
  AppendJsonFloatField(out, "postAttentionMean", post_attention_stats.mean,
                       first);
  AppendJsonFloatField(out, "postAttentionL2Norm",
                       post_attention_stats.l2_norm, first);
  AppendJsonFloatField(out, "postAttentionChecksum",
                       post_attention_stats.checksum, first);
  AppendJsonFloatField(out, "gateMin", gate_stats.min, first);
  AppendJsonFloatField(out, "gateMax", gate_stats.max, first);
  AppendJsonFloatField(out, "gateMean", gate_stats.mean, first);
  AppendJsonFloatField(out, "gateL2Norm", gate_stats.l2_norm, first);
  AppendJsonFloatField(out, "gateChecksum", gate_stats.checksum, first);
  AppendJsonFloatField(out, "upMin", up_stats.min, first);
  AppendJsonFloatField(out, "upMax", up_stats.max, first);
  AppendJsonFloatField(out, "upMean", up_stats.mean, first);
  AppendJsonFloatField(out, "upL2Norm", up_stats.l2_norm, first);
  AppendJsonFloatField(out, "upChecksum", up_stats.checksum, first);
  AppendJsonFloatField(out, "ffnHiddenMin", ffn_hidden_stats.min, first);
  AppendJsonFloatField(out, "ffnHiddenMax", ffn_hidden_stats.max, first);
  AppendJsonFloatField(out, "ffnHiddenMean", ffn_hidden_stats.mean, first);
  AppendJsonFloatField(out, "ffnHiddenL2Norm", ffn_hidden_stats.l2_norm,
                       first);
  AppendJsonFloatField(out, "ffnHiddenChecksum", ffn_hidden_stats.checksum,
                       first);
  AppendJsonFloatField(out, "ffnOutputMin", ffn_output_stats.min, first);
  AppendJsonFloatField(out, "ffnOutputMax", ffn_output_stats.max, first);
  AppendJsonFloatField(out, "ffnOutputMean", ffn_output_stats.mean, first);
  AppendJsonFloatField(out, "ffnOutputL2Norm", ffn_output_stats.l2_norm, first);
  AppendJsonFloatField(out, "ffnOutputChecksum", ffn_output_stats.checksum,
                       first);
  AppendJsonFloatField(out, "outputMin", layer_output_stats.min, first);
  AppendJsonFloatField(out, "outputMax", layer_output_stats.max, first);
  AppendJsonFloatField(out, "outputMean", layer_output_stats.mean, first);
  AppendJsonFloatField(out, "outputL2Norm", layer_output_stats.l2_norm, first);
  AppendJsonFloatField(out, "outputChecksum", layer_output_stats.checksum,
                       first);

  AppendJsonIntArrayField(out, "tokenIds", token_ids, first);
  AppendJsonIntArrayField(out, "qHeadToKvHead", q_head_to_kv_head, first);
  AppendJsonFloatArrayField(out, "queryValues", query_full,
                            returned_attention_values, first);
  AppendJsonFloatArrayField(out, "attentionValues", attention_output,
                            returned_attention_values, first);
  AppendJsonFloatArrayField(out, "attentionProjectedValues",
                            attention_projection, returned_output_values,
                            first);
  AppendJsonFloatArrayField(out, "postAttentionValues", post_attention,
                            returned_output_values, first);
  AppendJsonFloatArrayField(out, "gateValues", gate, returned_ffn_values,
                            first);
  AppendJsonFloatArrayField(out, "upValues", up, returned_ffn_values, first);
  AppendJsonFloatArrayField(out, "ffnHiddenValues", ffn_hidden,
                            returned_ffn_values, first);
  AppendJsonFloatArrayField(out, "ffnOutputValues", ffn_output,
                            returned_output_values, first);
  AppendJsonFloatArrayField(out, "outputValues", layer_output,
                            returned_output_values, first);

  out << "," << Quote("firstHeadScores") << ":[";
  for (size_t i = 0; i < returned_score_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(first_head_scores[i])) {
      out << std::setprecision(9) << first_head_scores[i];
    } else {
      out << "null";
    }
  }
  out << "]";

  out << "," << Quote("firstHeadProbabilities") << ":[";
  for (size_t i = 0; i < returned_score_values; ++i) {
    if (i != 0) out << ",";
    if (std::isfinite(first_head_probabilities[i])) {
      out << std::setprecision(9) << first_head_probabilities[i];
    } else {
      out << "null";
    }
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string TransformerStack(
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
    double rope_theta, int max_values) {
  if (token_ids.empty()) {
    throw std::runtime_error("Transformer stack tokenIds must be non-empty");
  }
  if (token_ids.size() > 512) {
    throw std::runtime_error(
        "Transformer stack diagnostic sequence is too large");
  }
  if (start_position < 0) {
    throw std::runtime_error(
        "Transformer stack startPosition must be non-negative");
  }
  if (head_dim <= 0) {
    throw std::runtime_error("Transformer stack headDim must be positive");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0) {
      throw std::runtime_error(
          "Transformer stack token ids must be non-negative");
    }
  }

  int read_index = read_position;
  if (read_index < 0) {
    read_index = static_cast<int>(token_ids.size() - 1);
  }
  if (read_index < 0 ||
      static_cast<uint64_t>(read_index) >=
          static_cast<uint64_t>(token_ids.size())) {
    throw std::runtime_error(
        "Transformer stack readPosition is outside the sequence");
  }
  if (start_position >
      std::numeric_limits<int64_t>::max() -
          static_cast<int64_t>(token_ids.size() - 1)) {
    throw std::runtime_error("Transformer stack cache positions overflow");
  }

  const GgufIndex index = ReadGgufIndex(path);
  const TensorInfo* embedding = FindTensor(index, "token_embd.weight");
  if (embedding == nullptr) {
    throw std::runtime_error("GGUF tensor token_embd.weight was not found");
  }
  if (!embedding->has_known_byte_size) {
    throw std::runtime_error(
        "token_embd.weight uses an unsupported GGML tensor type");
  }
  if (embedding->shape.size() < 2) {
    throw std::runtime_error("token_embd.weight must be a 2D tensor");
  }
  const uint64_t hidden_size = embedding->shape[0];
  const uint64_t vocab_size = embedding->shape[1];
  if (hidden_size == 0 || vocab_size == 0) {
    throw std::runtime_error("token_embd.weight has an invalid shape");
  }
  for (const int token_id : token_ids) {
    if (static_cast<uint64_t>(token_id) >= vocab_size) {
      throw std::runtime_error("Token id is outside token_embd.weight vocab");
    }
  }

  const uint64_t block_count = ParseUnsigned(
      index.metadata, index.arch_prefix + "block_count", 0);
  int resolved_layer_count = layer_count;
  if (resolved_layer_count <= 0) {
    if (block_count == 0 ||
        block_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "Transformer stack layerCount must be provided for this GGUF");
    }
    resolved_layer_count = static_cast<int>(block_count);
  }
  if (resolved_layer_count <= 0 || resolved_layer_count > 128) {
    throw std::runtime_error(
        "Transformer stack layerCount must be between 1 and 128");
  }
  if (block_count != 0 &&
      static_cast<uint64_t>(resolved_layer_count) > block_count) {
    throw std::runtime_error(
        "Transformer stack layerCount exceeds GGUF block_count");
  }

  double resolved_rope_theta = rope_theta;
  if (!(resolved_rope_theta > 0.0) || !std::isfinite(resolved_rope_theta)) {
    resolved_rope_theta =
        ParseDouble(index.metadata, index.arch_prefix + "rope.freq_base",
                    10000.0);
  }

  FileDescriptor fd(path);
  std::vector<std::vector<float>> hidden_states;
  hidden_states.reserve(token_ids.size());
  for (const int token_id : token_ids) {
    hidden_states.push_back(DecodeTensorRow(
        fd.get(), *embedding, static_cast<uint64_t>(token_id), hidden_size));
  }

  DecoderLayerRun last_layer;
  std::vector<double> layer_output_checksums;
  layer_output_checksums.reserve(static_cast<size_t>(resolved_layer_count));
  uint64_t kv_cache_element_count = 0;
  for (int layer_index = 0; layer_index < resolved_layer_count; ++layer_index) {
    last_layer = RunDecoderLayer(
        fd.get(), index, hidden_states, layer_index, read_index,
        attn_norm_tensor_suffix, query_weight_tensor_suffix,
        key_weight_tensor_suffix, value_weight_tensor_suffix,
        output_weight_tensor_suffix, ffn_norm_tensor_suffix,
        gate_weight_tensor_suffix, up_weight_tensor_suffix,
        down_weight_tensor_suffix, epsilon, start_position,
        static_cast<uint64_t>(head_dim), resolved_rope_theta);
    hidden_states = last_layer.hidden_states;
    const VectorStats selected_stats =
        ComputeStats(hidden_states[static_cast<size_t>(read_index)]);
    layer_output_checksums.push_back(selected_stats.checksum);
    kv_cache_element_count =
        CheckedAdd(kv_cache_element_count, last_layer.kv_cache_element_count);
  }

  const std::vector<float>& selected_output =
      hidden_states[static_cast<size_t>(read_index)];
  const VectorStats output_stats = ComputeStats(selected_output);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_output_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       selected_output.size());
  const size_t returned_attention_values = std::min<size_t>(
      static_cast<size_t>(preview_count), last_layer.attention_values.size());
  const size_t returned_ffn_values = std::min<size_t>(
      static_cast<size_t>(preview_count), last_layer.ffn_hidden_values.size());
  const size_t returned_score_values = std::min<size_t>(
      static_cast<size_t>(preview_count), last_layer.first_head_scores.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "attnNormTensorSuffix", attn_norm_tensor_suffix, first);
  AppendJsonField(out, "queryWeightTensorSuffix", query_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "keyWeightTensorSuffix", key_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "valueWeightTensorSuffix", value_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "outputWeightTensorSuffix", output_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "ffnNormTensorSuffix", ffn_norm_tensor_suffix, first);
  AppendJsonField(out, "gateWeightTensorSuffix", gate_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "upWeightTensorSuffix", up_weight_tensor_suffix, first);
  AppendJsonField(out, "downWeightTensorSuffix", down_weight_tensor_suffix,
                  first);
  AppendJsonNumberField(out, "blockCount", block_count, first);
  AppendJsonNumberField(out, "layerCount",
                        static_cast<uint64_t>(resolved_layer_count), first);
  AppendJsonNumberField(out, "layersVisited",
                        static_cast<uint64_t>(resolved_layer_count), first);
  AppendJsonNumberField(out, "tensorsVisited",
                        static_cast<uint64_t>(resolved_layer_count) * 9,
                        first);
  AppendJsonNumberField(out, "sequenceLength",
                        static_cast<uint64_t>(token_ids.size()), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(read_index), first);
  AppendJsonNumberField(
      out, "selectedTokenId",
      static_cast<uint64_t>(token_ids[static_cast<size_t>(read_index)]), first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "hiddenSize", hidden_size, first);
  AppendJsonNumberField(out, "vocabSize", vocab_size, first);
  AppendJsonNumberField(out, "queryWidth", last_layer.query_width, first);
  AppendJsonNumberField(out, "keyWidth", last_layer.key_width, first);
  AppendJsonNumberField(out, "valueWidth", last_layer.value_width, first);
  AppendJsonNumberField(out, "ffnWidth", last_layer.ffn_width, first);
  AppendJsonNumberField(out, "headDim", static_cast<uint64_t>(head_dim),
                        first);
  AppendJsonNumberField(out, "queryHeadCount", last_layer.query_head_count,
                        first);
  AppendJsonNumberField(out, "kvHeadCount", last_layer.kv_head_count, first);
  AppendJsonNumberField(out, "groupSize", last_layer.group_size, first);
  AppendJsonNumberField(out, "kvCacheLayerCount",
                        static_cast<uint64_t>(resolved_layer_count), first);
  AppendJsonNumberField(out, "kvCacheElementCount", kv_cache_element_count,
                        first);
  AppendJsonNumberField(out, "kvCacheBytesFp32",
                        CheckedMultiply(kv_cache_element_count, 4), first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedOutputValueCount",
                        static_cast<uint64_t>(returned_output_values), first);
  AppendJsonNumberField(out, "returnedAttentionValueCount",
                        static_cast<uint64_t>(returned_attention_values),
                        first);
  AppendJsonNumberField(out, "returnedFfnValueCount",
                        static_cast<uint64_t>(returned_ffn_values), first);
  AppendJsonNumberField(out, "returnedScoreValueCount",
                        static_cast<uint64_t>(returned_score_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "ropeTheta", resolved_rope_theta, first);
  AppendJsonFloatField(out, "scale",
                       1.0 / std::sqrt(static_cast<double>(head_dim)), first);
  AppendJsonFloatField(out, "lastLayerAttnMeanSquare",
                       last_layer.attn_mean_square, first);
  AppendJsonFloatField(out, "lastLayerAttnInvRms", last_layer.attn_inv_rms,
                       first);
  AppendJsonFloatField(out, "lastLayerFfnMeanSquare",
                       last_layer.ffn_mean_square, first);
  AppendJsonFloatField(out, "lastLayerFfnInvRms", last_layer.ffn_inv_rms,
                       first);
  AppendJsonFloatField(out, "outputMin", output_stats.min, first);
  AppendJsonFloatField(out, "outputMax", output_stats.max, first);
  AppendJsonFloatField(out, "outputMean", output_stats.mean, first);
  AppendJsonFloatField(out, "outputL2Norm", output_stats.l2_norm, first);
  AppendJsonFloatField(out, "outputChecksum", output_stats.checksum, first);

  AppendJsonIntArrayField(out, "tokenIds", token_ids, first);
  AppendJsonIntArrayField(out, "qHeadToKvHead", last_layer.q_head_to_kv_head,
                          first);
  AppendJsonDoubleArrayField(out, "layerOutputChecksums",
                             layer_output_checksums,
                             layer_output_checksums.size(), first);
  AppendJsonFloatArrayField(out, "lastLayerQueryValues",
                            last_layer.query_values, returned_attention_values,
                            first);
  AppendJsonFloatArrayField(out, "lastLayerAttentionValues",
                            last_layer.attention_values,
                            returned_attention_values, first);
  AppendJsonFloatArrayField(out, "lastLayerAttentionProjectedValues",
                            last_layer.attention_projected_values,
                            returned_output_values, first);
  AppendJsonFloatArrayField(out, "lastLayerPostAttentionValues",
                            last_layer.post_attention_values,
                            returned_output_values, first);
  AppendJsonFloatArrayField(out, "lastLayerGateValues",
                            last_layer.gate_values, returned_ffn_values, first);
  AppendJsonFloatArrayField(out, "lastLayerUpValues", last_layer.up_values,
                            returned_ffn_values, first);
  AppendJsonFloatArrayField(out, "lastLayerFfnHiddenValues",
                            last_layer.ffn_hidden_values, returned_ffn_values,
                            first);
  AppendJsonFloatArrayField(out, "lastLayerFfnOutputValues",
                            last_layer.ffn_output_values,
                            returned_output_values, first);
  AppendJsonFloatArrayField(out, "outputValues", selected_output,
                            returned_output_values, first);
  AppendJsonDoubleArrayField(out, "lastLayerFirstHeadScores",
                             last_layer.first_head_scores,
                             returned_score_values, first);
  AppendJsonDoubleArrayField(out, "lastLayerFirstHeadProbabilities",
                             last_layer.first_head_probabilities,
                             returned_score_values, first);
  out << "}";
  return out.str();
}

std::string GenerateNextToken(
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
    double rope_theta, int top_k, int max_values) {
  if (final_norm_tensor_name.empty() || lm_head_tensor_name.empty()) {
    throw std::runtime_error(
        "Generate next-token final norm and lm_head tensor names are required");
  }

  const TransformerStackRun stack = RunTransformerStack(
      path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
      key_weight_tensor_suffix, value_weight_tensor_suffix,
      output_weight_tensor_suffix, ffn_norm_tensor_suffix,
      gate_weight_tensor_suffix, up_weight_tensor_suffix,
      down_weight_tensor_suffix, epsilon, start_position, read_position,
      layer_count, head_dim, rope_theta);

  const TensorInfo* final_norm = RequireKnownTensor(
      stack.index, final_norm_tensor_name, "generate final norm");
  const TensorInfo* lm_head = RequireKnownTensor(
      stack.index, lm_head_tensor_name, "generate lm_head");
  if (final_norm->element_count != stack.hidden_size) {
    throw std::runtime_error(
        "Generate final norm tensor length does not match hidden size");
  }
  if (lm_head->shape.size() < 2) {
    throw std::runtime_error("Generate lm_head tensor must be 2D");
  }
  const uint64_t lm_head_input_length = lm_head->shape[0];
  const uint64_t logit_count = lm_head->shape[1];
  if (lm_head_input_length != stack.hidden_size || logit_count == 0) {
    throw std::runtime_error("Generate lm_head tensor shape is incompatible");
  }
  if (logit_count >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Generate lm_head vocab is too large");
  }

  FileDescriptor fd(path);
  const std::vector<float> final_norm_weights =
      DecodeTensorVector(fd.get(), *final_norm);
  const RmsNormOutput normalized =
      ApplyRmsNorm(stack.selected_output, final_norm_weights, epsilon);
  const std::vector<float> logits =
      ProjectVectorRows(fd.get(), *lm_head, normalized.values);
  if (logits.empty()) {
    throw std::runtime_error("Generate lm_head produced no logits");
  }

  int next_token_id = 0;
  float next_token_logit = logits.front();
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > next_token_logit) {
      next_token_logit = logits[i];
      next_token_id = static_cast<int>(i);
    }
  }

  int resolved_top_k = top_k;
  if (resolved_top_k < 0) resolved_top_k = 0;
  resolved_top_k = std::min<int>(resolved_top_k, 128);
  resolved_top_k =
      std::min<int>(resolved_top_k, static_cast<int>(logits.size()));
  std::vector<int> ranked_ids;
  ranked_ids.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    ranked_ids.push_back(static_cast<int>(i));
  }
  std::partial_sort(
      ranked_ids.begin(), ranked_ids.begin() + resolved_top_k,
      ranked_ids.end(), [&logits](int left, int right) {
        if (logits[static_cast<size_t>(left)] ==
            logits[static_cast<size_t>(right)]) {
          return left < right;
        }
        return logits[static_cast<size_t>(left)] >
               logits[static_cast<size_t>(right)];
      });
  ranked_ids.resize(static_cast<size_t>(resolved_top_k));
  std::vector<float> top_logits;
  top_logits.reserve(ranked_ids.size());
  for (const int token_id : ranked_ids) {
    top_logits.push_back(logits[static_cast<size_t>(token_id)]);
  }

  const VectorStats hidden_stats = ComputeStats(stack.selected_output);
  const VectorStats normalized_stats = ComputeStats(normalized.values);
  const VectorStats logit_stats = ComputeStats(logits);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_hidden_values = std::min<size_t>(
      static_cast<size_t>(preview_count), stack.selected_output.size());
  const size_t returned_normalized_values = std::min<size_t>(
      static_cast<size_t>(preview_count), normalized.values.size());
  const size_t returned_logit_values =
      std::min<size_t>(static_cast<size_t>(preview_count), logits.size());

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", stack.index.path, first);
  AppendJsonField(out, "architecture", stack.index.architecture, first);
  AppendJsonField(out, "embeddingTensorName", stack.embedding_tensor_name,
                  first);
  AppendJsonField(out, "embeddingTensorType",
                  TensorTypeName(stack.embedding_tensor_type), first);
  AppendJsonField(out, "finalNormTensorName", final_norm->name, first);
  AppendJsonField(out, "finalNormTensorType", TensorTypeName(final_norm->type),
                  first);
  AppendJsonField(out, "lmHeadTensorName", lm_head->name, first);
  AppendJsonField(out, "lmHeadTensorType", TensorTypeName(lm_head->type),
                  first);
  AppendJsonField(out, "attnNormTensorSuffix", attn_norm_tensor_suffix, first);
  AppendJsonField(out, "queryWeightTensorSuffix", query_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "keyWeightTensorSuffix", key_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "valueWeightTensorSuffix", value_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "outputWeightTensorSuffix", output_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "ffnNormTensorSuffix", ffn_norm_tensor_suffix, first);
  AppendJsonField(out, "gateWeightTensorSuffix", gate_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "upWeightTensorSuffix", up_weight_tensor_suffix, first);
  AppendJsonField(out, "downWeightTensorSuffix", down_weight_tensor_suffix,
                  first);
  AppendJsonNumberField(out, "blockCount", stack.block_count, first);
  AppendJsonNumberField(out, "layerCount",
                        static_cast<uint64_t>(stack.resolved_layer_count),
                        first);
  AppendJsonNumberField(out, "layersVisited",
                        static_cast<uint64_t>(stack.resolved_layer_count),
                        first);
  AppendJsonNumberField(
      out, "tensorsVisited",
      static_cast<uint64_t>(stack.resolved_layer_count) * 9 + 2, first);
  AppendJsonNumberField(out, "sequenceLength",
                        static_cast<uint64_t>(token_ids.size()), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(stack.read_index), first);
  AppendJsonNumberField(
      out, "selectedTokenId",
      static_cast<uint64_t>(token_ids[static_cast<size_t>(stack.read_index)]),
      first);
  AppendJsonNumberField(out, "nextTokenId",
                        static_cast<uint64_t>(next_token_id), first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "hiddenSize", stack.hidden_size, first);
  AppendJsonNumberField(out, "vocabSize", stack.vocab_size, first);
  AppendJsonNumberField(out, "lmHeadInputLength", lm_head_input_length, first);
  AppendJsonNumberField(out, "logitCount", logit_count, first);
  AppendJsonNumberField(out, "headDim", static_cast<uint64_t>(head_dim),
                        first);
  AppendJsonNumberField(out, "queryHeadCount",
                        stack.last_layer.query_head_count, first);
  AppendJsonNumberField(out, "kvHeadCount", stack.last_layer.kv_head_count,
                        first);
  AppendJsonNumberField(out, "groupSize", stack.last_layer.group_size, first);
  AppendJsonNumberField(out, "kvCacheLayerCount",
                        static_cast<uint64_t>(stack.resolved_layer_count),
                        first);
  AppendJsonNumberField(out, "kvCacheElementCount",
                        stack.kv_cache_element_count, first);
  AppendJsonNumberField(out, "kvCacheBytesFp32",
                        CheckedMultiply(stack.kv_cache_element_count, 4),
                        first);
  AppendJsonNumberField(out, "requestedTopK",
                        static_cast<uint64_t>(std::max(top_k, 0)), first);
  AppendJsonNumberField(out, "returnedTopK",
                        static_cast<uint64_t>(ranked_ids.size()), first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedHiddenValueCount",
                        static_cast<uint64_t>(returned_hidden_values), first);
  AppendJsonNumberField(out, "returnedNormalizedValueCount",
                        static_cast<uint64_t>(returned_normalized_values),
                        first);
  AppendJsonNumberField(out, "returnedLogitValueCount",
                        static_cast<uint64_t>(returned_logit_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "ropeTheta", stack.resolved_rope_theta, first);
  AppendJsonFloatField(out, "scale",
                       1.0 / std::sqrt(static_cast<double>(head_dim)), first);
  AppendJsonFloatField(out, "finalMeanSquare", normalized.mean_square, first);
  AppendJsonFloatField(out, "finalInvRms", normalized.inv_rms, first);
  AppendJsonFloatField(out, "nextTokenLogit", next_token_logit, first);
  AppendJsonFloatField(out, "hiddenMin", hidden_stats.min, first);
  AppendJsonFloatField(out, "hiddenMax", hidden_stats.max, first);
  AppendJsonFloatField(out, "hiddenMean", hidden_stats.mean, first);
  AppendJsonFloatField(out, "hiddenL2Norm", hidden_stats.l2_norm, first);
  AppendJsonFloatField(out, "hiddenChecksum", hidden_stats.checksum, first);
  AppendJsonFloatField(out, "normalizedMin", normalized_stats.min, first);
  AppendJsonFloatField(out, "normalizedMax", normalized_stats.max, first);
  AppendJsonFloatField(out, "normalizedMean", normalized_stats.mean, first);
  AppendJsonFloatField(out, "normalizedL2Norm", normalized_stats.l2_norm,
                       first);
  AppendJsonFloatField(out, "normalizedChecksum", normalized_stats.checksum,
                       first);
  AppendJsonFloatField(out, "logitMin", logit_stats.min, first);
  AppendJsonFloatField(out, "logitMax", logit_stats.max, first);
  AppendJsonFloatField(out, "logitMean", logit_stats.mean, first);
  AppendJsonFloatField(out, "logitL2Norm", logit_stats.l2_norm, first);
  AppendJsonFloatField(out, "logitChecksum", logit_stats.checksum, first);

  AppendJsonIntArrayField(out, "tokenIds", token_ids, first);
  AppendJsonIntArrayField(out, "qHeadToKvHead",
                          stack.last_layer.q_head_to_kv_head, first);
  AppendJsonDoubleArrayField(out, "layerOutputChecksums",
                             stack.layer_output_checksums,
                             stack.layer_output_checksums.size(), first);
  AppendJsonFloatArrayField(out, "hiddenValues", stack.selected_output,
                            returned_hidden_values, first);
  AppendJsonFloatArrayField(out, "normalizedValues", normalized.values,
                            returned_normalized_values, first);
  AppendJsonFloatArrayField(out, "logitValues", logits,
                            returned_logit_values, first);
  AppendJsonIntArrayField(out, "topTokenIds", ranked_ids, first);
  AppendJsonFloatArrayField(out, "topLogits", top_logits, top_logits.size(),
                            first);
  out << "}";
  return out.str();
}

std::string GenerateTokens(
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
    double rope_theta, int max_new_tokens, int top_k, int max_values) {
  if (final_norm_tensor_name.empty() || lm_head_tensor_name.empty()) {
    throw std::runtime_error(
        "Generate tokens final norm and lm_head tensor names are required");
  }
  if (max_new_tokens <= 0 || max_new_tokens > 32) {
    throw std::runtime_error(
        "Generate tokens maxNewTokens must be between 1 and 32");
  }

  TransformerStackRun stack = RunTransformerStack(
      path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
      key_weight_tensor_suffix, value_weight_tensor_suffix,
      output_weight_tensor_suffix, ffn_norm_tensor_suffix,
      gate_weight_tensor_suffix, up_weight_tensor_suffix,
      down_weight_tensor_suffix, epsilon, start_position, read_position,
      layer_count, head_dim, rope_theta);
  if (stack.read_index != static_cast<int>(token_ids.size() - 1)) {
    throw std::runtime_error(
        "Generate tokens readPosition must resolve to the last prompt token");
  }
  if (stack.layer_caches.size() !=
      static_cast<size_t>(stack.resolved_layer_count)) {
    throw std::runtime_error("Generate tokens layer cache count is invalid");
  }

  const TensorInfo* embedding =
      RequireKnownTensor(stack.index, "token_embd.weight",
                         "generate token embedding");
  const TensorInfo* final_norm = RequireKnownTensor(
      stack.index, final_norm_tensor_name, "generate final norm");
  const TensorInfo* lm_head = RequireKnownTensor(
      stack.index, lm_head_tensor_name, "generate lm_head");
  if (embedding->shape.size() < 2 || embedding->shape[0] != stack.hidden_size ||
      embedding->shape[1] != stack.vocab_size) {
    throw std::runtime_error("Generate token embedding shape is incompatible");
  }
  if (lm_head->shape.size() < 2 || lm_head->shape[0] != stack.hidden_size ||
      lm_head->shape[1] != stack.vocab_size) {
    throw std::runtime_error(
        "Generate tokens requires lm_head vocab to match token embeddings");
  }

  FileDescriptor fd(path);
  std::vector<float> current_hidden = stack.selected_output;
  std::vector<int> generated_token_ids;
  generated_token_ids.reserve(static_cast<size_t>(max_new_tokens));
  std::vector<float> generated_token_logits;
  generated_token_logits.reserve(static_cast<size_t>(max_new_tokens));
  std::vector<int> all_token_ids = token_ids;
  all_token_ids.reserve(token_ids.size() + static_cast<size_t>(max_new_tokens));

  NextTokenProjection projection;
  int64_t next_absolute_position =
      start_position + static_cast<int64_t>(token_ids.size());
  for (int step = 0; step < max_new_tokens; ++step) {
    projection = ProjectNextToken(fd.get(), *final_norm, *lm_head,
                                  current_hidden, epsilon, top_k);
    if (projection.next_token_id < 0 ||
        static_cast<uint64_t>(projection.next_token_id) >= stack.vocab_size) {
      throw std::runtime_error(
          "Generate tokens sampled token is outside token embeddings");
    }
    generated_token_ids.push_back(projection.next_token_id);
    generated_token_logits.push_back(projection.next_token_logit);
    all_token_ids.push_back(projection.next_token_id);

    if (step + 1 == max_new_tokens) {
      break;
    }

    current_hidden = DecodeTensorRow(
        fd.get(), *embedding, static_cast<uint64_t>(projection.next_token_id),
        stack.hidden_size);
    for (int layer_index = 0; layer_index < stack.resolved_layer_count;
         ++layer_index) {
      current_hidden = RunDecoderLayerToken(
          fd.get(), stack.index, current_hidden, layer_index,
          next_absolute_position,
          stack.layer_caches[static_cast<size_t>(layer_index)],
          attn_norm_tensor_suffix, query_weight_tensor_suffix,
          key_weight_tensor_suffix, value_weight_tensor_suffix,
          output_weight_tensor_suffix, ffn_norm_tensor_suffix,
          gate_weight_tensor_suffix, up_weight_tensor_suffix,
          down_weight_tensor_suffix, epsilon, static_cast<uint64_t>(head_dim),
          stack.resolved_rope_theta);
    }
    next_absolute_position += 1;
  }

  uint64_t final_kv_cache_element_count = 0;
  for (const LayerKvCache& cache : stack.layer_caches) {
    final_kv_cache_element_count = CheckedAdd(
        final_kv_cache_element_count,
        static_cast<uint64_t>(cache.key_cache.size()));
    final_kv_cache_element_count = CheckedAdd(
        final_kv_cache_element_count,
        static_cast<uint64_t>(cache.value_cache.size()));
  }

  const VectorStats hidden_stats = ComputeStats(current_hidden);
  const VectorStats normalized_stats = ComputeStats(projection.normalized.values);
  const VectorStats logit_stats = ComputeStats(projection.logits);
  int preview_count = max_values;
  if (preview_count < 0) preview_count = 0;
  preview_count = std::min<int>(preview_count, 4096);
  const size_t returned_hidden_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       current_hidden.size());
  const size_t returned_normalized_values = std::min<size_t>(
      static_cast<size_t>(preview_count), projection.normalized.values.size());
  const size_t returned_logit_values =
      std::min<size_t>(static_cast<size_t>(preview_count),
                       projection.logits.size());
  const uint64_t incremental_layer_steps =
      static_cast<uint64_t>(std::max(max_new_tokens - 1, 0)) *
      static_cast<uint64_t>(stack.resolved_layer_count);
  const uint64_t layers_visited =
      static_cast<uint64_t>(stack.resolved_layer_count) +
      incremental_layer_steps;
  const uint64_t tensors_visited =
      static_cast<uint64_t>(stack.resolved_layer_count) * 9 +
      incremental_layer_steps * 9 +
      static_cast<uint64_t>(max_new_tokens) * 2;
  const uint64_t final_cache_token_count =
      stack.layer_caches.empty() ? 0 : stack.layer_caches.front().token_count;

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", stack.index.path, first);
  AppendJsonField(out, "architecture", stack.index.architecture, first);
  AppendJsonField(out, "embeddingTensorName", embedding->name, first);
  AppendJsonField(out, "embeddingTensorType", TensorTypeName(embedding->type),
                  first);
  AppendJsonField(out, "finalNormTensorName", final_norm->name, first);
  AppendJsonField(out, "finalNormTensorType", TensorTypeName(final_norm->type),
                  first);
  AppendJsonField(out, "lmHeadTensorName", lm_head->name, first);
  AppendJsonField(out, "lmHeadTensorType", TensorTypeName(lm_head->type),
                  first);
  AppendJsonField(out, "attnNormTensorSuffix", attn_norm_tensor_suffix, first);
  AppendJsonField(out, "queryWeightTensorSuffix", query_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "keyWeightTensorSuffix", key_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "valueWeightTensorSuffix", value_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "outputWeightTensorSuffix", output_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "ffnNormTensorSuffix", ffn_norm_tensor_suffix, first);
  AppendJsonField(out, "gateWeightTensorSuffix", gate_weight_tensor_suffix,
                  first);
  AppendJsonField(out, "upWeightTensorSuffix", up_weight_tensor_suffix, first);
  AppendJsonField(out, "downWeightTensorSuffix", down_weight_tensor_suffix,
                  first);
  AppendJsonBoolField(out, "cacheReused", true, first);
  AppendJsonNumberField(out, "blockCount", stack.block_count, first);
  AppendJsonNumberField(out, "layerCount",
                        static_cast<uint64_t>(stack.resolved_layer_count),
                        first);
  AppendJsonNumberField(out, "layersVisited", layers_visited, first);
  AppendJsonNumberField(out, "tensorsVisited", tensors_visited, first);
  AppendJsonNumberField(out, "sequenceLength",
                        static_cast<uint64_t>(token_ids.size()), first);
  AppendJsonNumberField(out, "generatedTokenCount",
                        static_cast<uint64_t>(generated_token_ids.size()),
                        first);
  AppendJsonNumberField(out, "totalTokenCount",
                        static_cast<uint64_t>(all_token_ids.size()), first);
  AppendJsonNumberField(out, "maxNewTokens",
                        static_cast<uint64_t>(max_new_tokens), first);
  AppendJsonNumberField(out, "readPosition",
                        static_cast<uint64_t>(stack.read_index), first);
  AppendJsonNumberField(
      out, "selectedTokenId",
      static_cast<uint64_t>(token_ids[static_cast<size_t>(stack.read_index)]),
      first);
  AppendJsonNumberField(
      out, "firstGeneratedTokenId",
      static_cast<uint64_t>(generated_token_ids.front()), first);
  AppendJsonNumberField(
      out, "lastGeneratedTokenId",
      static_cast<uint64_t>(generated_token_ids.back()), first);
  AppendJsonNumberField(out, "startPosition",
                        static_cast<uint64_t>(start_position), first);
  AppendJsonNumberField(out, "hiddenSize", stack.hidden_size, first);
  AppendJsonNumberField(out, "vocabSize", stack.vocab_size, first);
  AppendJsonNumberField(out, "lmHeadInputLength", lm_head->shape[0], first);
  AppendJsonNumberField(out, "logitCount", lm_head->shape[1], first);
  AppendJsonNumberField(out, "headDim", static_cast<uint64_t>(head_dim),
                        first);
  AppendJsonNumberField(out, "queryHeadCount",
                        stack.last_layer.query_head_count, first);
  AppendJsonNumberField(out, "kvHeadCount", stack.last_layer.kv_head_count,
                        first);
  AppendJsonNumberField(out, "groupSize", stack.last_layer.group_size, first);
  AppendJsonNumberField(out, "kvCacheLayerCount",
                        static_cast<uint64_t>(stack.resolved_layer_count),
                        first);
  AppendJsonNumberField(out, "initialKvCacheTokenCount",
                        static_cast<uint64_t>(token_ids.size()), first);
  AppendJsonNumberField(out, "finalKvCacheTokenCount",
                        final_cache_token_count, first);
  AppendJsonNumberField(out, "initialKvCacheElementCount",
                        stack.kv_cache_element_count, first);
  AppendJsonNumberField(out, "finalKvCacheElementCount",
                        final_kv_cache_element_count, first);
  AppendJsonNumberField(out, "initialKvCacheBytesFp32",
                        CheckedMultiply(stack.kv_cache_element_count, 4),
                        first);
  AppendJsonNumberField(out, "finalKvCacheBytesFp32",
                        CheckedMultiply(final_kv_cache_element_count, 4),
                        first);
  AppendJsonNumberField(out, "requestedTopK",
                        static_cast<uint64_t>(std::max(top_k, 0)), first);
  AppendJsonNumberField(out, "returnedTopK",
                        static_cast<uint64_t>(projection.top_token_ids.size()),
                        first);
  AppendJsonNumberField(out, "requestedValueCount",
                        static_cast<uint64_t>(std::max(max_values, 0)), first);
  AppendJsonNumberField(out, "returnedHiddenValueCount",
                        static_cast<uint64_t>(returned_hidden_values), first);
  AppendJsonNumberField(out, "returnedNormalizedValueCount",
                        static_cast<uint64_t>(returned_normalized_values),
                        first);
  AppendJsonNumberField(out, "returnedLogitValueCount",
                        static_cast<uint64_t>(returned_logit_values), first);
  AppendJsonFloatField(out, "epsilon", epsilon, first);
  AppendJsonFloatField(out, "ropeTheta", stack.resolved_rope_theta, first);
  AppendJsonFloatField(out, "scale",
                       1.0 / std::sqrt(static_cast<double>(head_dim)), first);
  AppendJsonFloatField(out, "finalMeanSquare",
                       projection.normalized.mean_square, first);
  AppendJsonFloatField(out, "finalInvRms", projection.normalized.inv_rms,
                       first);
  AppendJsonFloatField(out, "lastTokenLogit", projection.next_token_logit,
                       first);
  AppendJsonFloatField(out, "hiddenMin", hidden_stats.min, first);
  AppendJsonFloatField(out, "hiddenMax", hidden_stats.max, first);
  AppendJsonFloatField(out, "hiddenMean", hidden_stats.mean, first);
  AppendJsonFloatField(out, "hiddenL2Norm", hidden_stats.l2_norm, first);
  AppendJsonFloatField(out, "hiddenChecksum", hidden_stats.checksum, first);
  AppendJsonFloatField(out, "normalizedMin", normalized_stats.min, first);
  AppendJsonFloatField(out, "normalizedMax", normalized_stats.max, first);
  AppendJsonFloatField(out, "normalizedMean", normalized_stats.mean, first);
  AppendJsonFloatField(out, "normalizedL2Norm", normalized_stats.l2_norm,
                       first);
  AppendJsonFloatField(out, "normalizedChecksum", normalized_stats.checksum,
                       first);
  AppendJsonFloatField(out, "logitMin", logit_stats.min, first);
  AppendJsonFloatField(out, "logitMax", logit_stats.max, first);
  AppendJsonFloatField(out, "logitMean", logit_stats.mean, first);
  AppendJsonFloatField(out, "logitL2Norm", logit_stats.l2_norm, first);
  AppendJsonFloatField(out, "logitChecksum", logit_stats.checksum, first);

  AppendJsonIntArrayField(out, "tokenIds", token_ids, first);
  AppendJsonIntArrayField(out, "generatedTokenIds", generated_token_ids,
                          first);
  AppendJsonIntArrayField(out, "allTokenIds", all_token_ids, first);
  AppendJsonFloatArrayField(out, "generatedTokenLogits",
                            generated_token_logits,
                            generated_token_logits.size(), first);
  AppendJsonIntArrayField(out, "qHeadToKvHead",
                          stack.last_layer.q_head_to_kv_head, first);
  AppendJsonDoubleArrayField(out, "layerOutputChecksums",
                             stack.layer_output_checksums,
                             stack.layer_output_checksums.size(), first);
  AppendJsonFloatArrayField(out, "hiddenValues", current_hidden,
                            returned_hidden_values, first);
  AppendJsonFloatArrayField(out, "normalizedValues",
                            projection.normalized.values,
                            returned_normalized_values, first);
  AppendJsonFloatArrayField(out, "logitValues", projection.logits,
                            returned_logit_values, first);
  AppendJsonIntArrayField(out, "topTokenIds", projection.top_token_ids, first);
  AppendJsonFloatArrayField(out, "topLogits", projection.top_logits,
                            projection.top_logits.size(), first);
  out << "}";
  return out.str();
}

std::string ValidateLayerStreaming(const std::string& path) {
  const GgufIndex index = ReadGgufIndex(path);
  std::map<int, std::vector<const TensorInfo*>> layers;
  std::vector<std::string> unknown_size_tensors;

  for (const TensorInfo& tensor : index.tensors) {
    const int layer = LayerIndexFromTensorName(tensor.name);
    if (layer < 0) continue;
    if (!tensor.has_known_byte_size) {
      unknown_size_tensors.push_back(tensor.name);
      continue;
    }
    layers[layer].push_back(&tensor);
  }

  if (!unknown_size_tensors.empty()) {
    throw std::runtime_error("Cannot validate layer streaming because a layer "
                             "tensor has an unsupported GGML type: " +
                             unknown_size_tensors.front());
  }
  if (layers.empty()) {
    throw std::runtime_error("No blk.N.* layer tensors found in GGUF file");
  }

  const uint64_t block_count = ParseUnsigned(
      index.metadata, index.arch_prefix + "block_count",
      static_cast<uint64_t>(layers.size()));
  std::vector<int> missing_layers;
  for (uint64_t layer = 0; layer < block_count; ++layer) {
    if (layers.find(static_cast<int>(layer)) == layers.end()) {
      missing_layers.push_back(static_cast<int>(layer));
    }
  }

  FileDescriptor fd(path);
  std::vector<LayerSummary> summaries;
  uint64_t tensors_visited = 0;
  uint64_t total_layer_span_bytes = 0;
  uint64_t max_layer_span_bytes = 0;
  uint64_t largest_tensor_bytes = 0;

  for (const auto& entry : layers) {
    const std::vector<const TensorInfo*>& tensors = entry.second;
    uint64_t start = std::numeric_limits<uint64_t>::max();
    uint64_t end = 0;
    uint64_t layer_largest_tensor = 0;

    for (const TensorInfo* tensor : tensors) {
      start = std::min(start, tensor->absolute_offset);
      const uint64_t tensor_end =
          CheckedAdd(tensor->absolute_offset, tensor->byte_size);
      end = std::max(end, tensor_end);
      layer_largest_tensor = std::max(layer_largest_tensor, tensor->byte_size);
    }

    if (start == std::numeric_limits<uint64_t>::max() || end <= start) {
      throw std::runtime_error("Invalid empty layer tensor span");
    }

    const uint64_t span = end - start;
    {
      MappedRegion mapped(fd.get(), start, span);
    }

    summaries.push_back(LayerSummary{
        entry.first,
        static_cast<uint64_t>(tensors.size()),
        start,
        span,
        layer_largest_tensor,
    });
    tensors_visited += tensors.size();
    total_layer_span_bytes = CheckedAdd(total_layer_span_bytes, span);
    max_layer_span_bytes = std::max(max_layer_span_bytes, span);
    largest_tensor_bytes = std::max(largest_tensor_bytes, layer_largest_tensor);
  }

  std::ostringstream out;
  out << "{";
  bool first = true;
  AppendJsonBoolField(out, "ok", true, first);
  AppendJsonField(out, "path", index.path, first);
  AppendJsonField(out, "architecture", index.architecture, first);
  AppendJsonNumberField(out, "fileSizeBytes", index.file_size, first);
  AppendJsonNumberField(out, "dataStartOffset", index.data_start, first);
  AppendJsonNumberField(out, "blockCount", block_count, first);
  AppendJsonNumberField(out, "layersDiscovered",
                        static_cast<uint64_t>(layers.size()), first);
  AppendJsonNumberField(out, "layersVisited",
                        static_cast<uint64_t>(summaries.size()), first);
  AppendJsonNumberField(out, "tensorsVisited", tensors_visited, first);
  AppendJsonNumberField(out, "maxLayerSpanBytes", max_layer_span_bytes, first);
  AppendJsonNumberField(out, "totalLayerSpanBytes", total_layer_span_bytes,
                        first);
  AppendJsonNumberField(out, "largestTensorBytes", largest_tensor_bytes, first);
  AppendJsonNumberField(out, "missingLayerCount",
                        static_cast<uint64_t>(missing_layers.size()), first);

  out << "," << Quote("missingLayers") << ":[";
  for (size_t i = 0; i < missing_layers.size(); ++i) {
    if (i != 0) out << ",";
    out << missing_layers[i];
  }
  out << "]";

  out << "," << Quote("layers") << ":[";
  for (size_t i = 0; i < summaries.size(); ++i) {
    const LayerSummary& layer = summaries[i];
    if (i != 0) out << ",";
    out << "{";
    bool layer_first = true;
    AppendJsonNumberField(out, "index", static_cast<uint64_t>(layer.index),
                          layer_first);
    AppendJsonNumberField(out, "tensorCount", layer.tensor_count, layer_first);
    AppendJsonNumberField(out, "spanStart", layer.span_start, layer_first);
    AppendJsonNumberField(out, "spanBytes", layer.span_bytes, layer_first);
    AppendJsonNumberField(out, "largestTensorBytes",
                          layer.largest_tensor_bytes, layer_first);
    AppendJsonBoolField(out, "mapped", true, layer_first);
    out << "}";
  }
  out << "]";
  out << "}";
  return out.str();
}

std::string ErrorJson(const std::string& error) {
  return std::string("{\"ok\":false,\"error\":") + Quote(error) + "}";
}

}  // namespace

std::string InspectGgufToJson(const std::string& path) noexcept {
  try {
    return InspectGguf(path);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF inspection error");
  }
}

std::string ValidateGgufLayerStreamingToJson(const std::string& path) noexcept {
  try {
    return ValidateLayerStreaming(path);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF layer streaming validation error");
  }
}

std::string ReadGgufTokenEmbeddingToJson(const std::string& path, int token_id,
                                         int max_values) noexcept {
  try {
    return ReadTokenEmbedding(path, token_id, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF token embedding read error");
  }
}

std::string RmsNormGgufTokenEmbeddingToJson(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    float epsilon, int max_values) noexcept {
  try {
    return RmsNormTokenEmbedding(path, token_id, norm_tensor_name, epsilon,
                                 max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF RMSNorm error");
  }
}

std::string MatVecGgufRmsNormTokenEmbeddingToJson(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    const std::string& weight_tensor_name, float epsilon,
    int max_values) noexcept {
  try {
    return MatVecRmsNormTokenEmbedding(path, token_id, norm_tensor_name,
                                       weight_tensor_name, epsilon,
                                       max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF MatVec error");
  }
}

std::string RopeGgufMatVecRmsNormTokenEmbeddingToJson(
    const std::string& path, int token_id, const std::string& norm_tensor_name,
    const std::string& weight_tensor_name, float epsilon, int64_t position,
    int head_dim, double rope_theta, int max_values) noexcept {
  try {
    return RopeMatVecRmsNormTokenEmbedding(
        path, token_id, norm_tensor_name, weight_tensor_name, epsilon, position,
        head_dim, rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF RoPE MatVec error");
  }
}

std::string KvCacheGgufRmsNormTokenEmbeddingsToJson(
    const std::string& path, const std::vector<int>& token_ids,
    const std::string& norm_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t read_position, int head_dim,
    double rope_theta, int max_values) noexcept {
  try {
    return KvCacheRmsNormTokenEmbeddings(
        path, token_ids, norm_tensor_name, key_weight_tensor_name,
        value_weight_tensor_name, epsilon, start_position, read_position,
        head_dim, rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF KV cache error");
  }
}

std::string AttentionGgufSingleHeadToJson(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, int head_index, double rope_theta,
    int max_values) noexcept {
  try {
    return AttentionSingleHead(path, token_ids, query_token_id,
                               norm_tensor_name, query_weight_tensor_name,
                               key_weight_tensor_name, value_weight_tensor_name,
                               epsilon, start_position, query_position,
                               read_position, attention_length, head_dim,
                               head_index, rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF single-head attention error");
  }
}

std::string AttentionGgufMultiHeadToJson(
    const std::string& path, const std::vector<int>& token_ids,
    int query_token_id, const std::string& norm_tensor_name,
    const std::string& query_weight_tensor_name,
    const std::string& key_weight_tensor_name,
    const std::string& value_weight_tensor_name, float epsilon,
    int64_t start_position, int64_t query_position, int read_position,
    int attention_length, int head_dim, double rope_theta,
    int max_values) noexcept {
  try {
    return AttentionMultiHead(path, token_ids, query_token_id,
                              norm_tensor_name, query_weight_tensor_name,
                              key_weight_tensor_name, value_weight_tensor_name,
                              epsilon, start_position, query_position,
                              read_position, attention_length, head_dim,
                              rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF multi-head attention error");
  }
}

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
    int max_values) noexcept {
  try {
    return TransformerLayer(
        path, token_ids, query_token_id, attn_norm_tensor_name,
        query_weight_tensor_name, key_weight_tensor_name,
        value_weight_tensor_name, output_weight_tensor_name,
        ffn_norm_tensor_name, gate_weight_tensor_name, up_weight_tensor_name,
        down_weight_tensor_name, epsilon, start_position, query_position,
        read_position, attention_length, head_dim, rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF transformer layer error");
  }
}

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
    double rope_theta, int max_values) noexcept {
  try {
    return TransformerStack(
        path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
        key_weight_tensor_suffix, value_weight_tensor_suffix,
        output_weight_tensor_suffix, ffn_norm_tensor_suffix,
        gate_weight_tensor_suffix, up_weight_tensor_suffix,
        down_weight_tensor_suffix, epsilon, start_position, read_position,
        layer_count, head_dim, rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF transformer stack error");
  }
}

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
    double rope_theta, int top_k, int max_values) noexcept {
  try {
    return GenerateNextToken(
        path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
        key_weight_tensor_suffix, value_weight_tensor_suffix,
        output_weight_tensor_suffix, ffn_norm_tensor_suffix,
        gate_weight_tensor_suffix, up_weight_tensor_suffix,
        down_weight_tensor_suffix, final_norm_tensor_name, lm_head_tensor_name,
        epsilon, start_position, read_position, layer_count, head_dim,
        rope_theta, top_k, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF next-token generation error");
  }
}

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
    int max_values) noexcept {
  try {
    return GenerateTokens(
        path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
        key_weight_tensor_suffix, value_weight_tensor_suffix,
        output_weight_tensor_suffix, ffn_norm_tensor_suffix,
        gate_weight_tensor_suffix, up_weight_tensor_suffix,
        down_weight_tensor_suffix, final_norm_tensor_name, lm_head_tensor_name,
        epsilon, start_position, read_position, layer_count, head_dim,
        rope_theta, max_new_tokens, top_k, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF greedy generation error");
  }
}

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
    double rope_theta, int max_values) noexcept {
  try {
    return CreateSession(
        path, token_ids, attn_norm_tensor_suffix, query_weight_tensor_suffix,
        key_weight_tensor_suffix, value_weight_tensor_suffix,
        output_weight_tensor_suffix, ffn_norm_tensor_suffix,
        gate_weight_tensor_suffix, up_weight_tensor_suffix,
        down_weight_tensor_suffix, final_norm_tensor_name, lm_head_tensor_name,
        epsilon, start_position, read_position, layer_count, head_dim,
        rope_theta, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF session creation error");
  }
}

std::string DecodeGgufSessionToJson(uint64_t session_id, int top_k,
                                     int max_values) noexcept {
  try {
    return DecodeSession(session_id, top_k, max_values);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF session decode error");
  }
}

std::string CloseGgufSessionToJson(uint64_t session_id) noexcept {
  try {
    return CloseSession(session_id);
  } catch (const std::exception& e) {
    return ErrorJson(e.what());
  } catch (...) {
    return ErrorJson("Unknown GGUF session close error");
  }
}

}  // namespace vaultiq
