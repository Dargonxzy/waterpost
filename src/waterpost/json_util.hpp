#pragma once

// 与 Python json 模块互通的 JSON：保留键的插入顺序（nlohmann::ordered_json），浮点按 Python repr 写出，
// 并像 Python 一样读写 NaN / Infinity / -Infinity（标准 JSON 不允许，Python 的清单里会出现）。
// 非有限浮点在内存里用哨兵字符串表示，jfloat() 生成、jnum() 读取，dump_python() 写回成裸记号。

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace ai3d::waterpost
{

using Json = nlohmann::ordered_json;

// 浮点值：有限值是 JSON 数字，NaN / ±Inf 是哨兵字符串。
Json jfloat(double value);

// 读数字：接受整数、浮点与非有限哨兵。类型不对时抛 WaterError。
double jnum(const Json& value);

// 读整数（接受整数或整值浮点）。
long long jint(const Json& value);

// 是否为数字（含非有限哨兵）。
bool jis_number(const Json& value);

// 对象里的数组成员（引用，不拷贝）；缺失或不是数组时返回空数组。
const Json& jarray(const Json& object, const char* key);

// 解析 Python json.dumps 写出的文本（允许 NaN / Infinity；允许 UTF-8 BOM）。
Json parse_python_json(const std::string& text);

// 读文件并解析（utf-8 / utf-8-sig）。
Json read_json_file(const std::filesystem::path& path);

// 按 Python json.dumps(ensure_ascii=False) 的格式写出：compact 为 separators=(",", ":")；
// 否则 indent=2（与 json.dumps(indent=2) 相同）。
std::string dump_python(const Json& value, bool compact);

// 原子写 JSON：先写 <path>.tmp 再替换（write_json_atomic）。
void write_json_atomic(const std::filesystem::path& path, const Json& value, bool compact);

// Python float repr（最短往返）。
std::string python_float_repr(double value);

} // namespace ai3d::waterpost
