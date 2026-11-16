#include "json_util.hpp"

#include "error.hpp"
#include "fs_util.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <system_error>

namespace ai3d::waterpost
{
namespace
{

const char* const kNanSentinel = "\001ai3d-json:NaN";
const char* const kPositiveInfinitySentinel = "\001ai3d-json:Infinity";
const char* const kNegativeInfinitySentinel = "\001ai3d-json:-Infinity";

void append_escaped(std::string& output, const std::string& text)
{
    output.push_back('"');
    for (unsigned char character : text)
    {
        switch (character)
        {
        case '"':
            output += "\\\"";
            break;
        case '\\':
            output += "\\\\";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        case '\b':
            output += "\\b";
            break;
        case '\f':
            output += "\\f";
            break;
        default:
            if (character < 0x20)
            {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
                output += buffer;
            }
            else
            {
                output.push_back(static_cast<char>(character));
            }
        }
    }
    output.push_back('"');
}

void dump_value(std::string& output, const Json& value, bool compact, int depth)
{
    const auto newline = [&](int level) {
        output.push_back('\n');
        output.append(static_cast<std::size_t>(level) * 2, ' ');
    };
    switch (value.type())
    {
    case Json::value_t::null:
        output += "null";
        return;
    case Json::value_t::boolean:
        output += value.get<bool>() ? "true" : "false";
        return;
    case Json::value_t::number_integer:
        output += std::to_string(value.get<long long>());
        return;
    case Json::value_t::number_unsigned:
        output += std::to_string(value.get<unsigned long long>());
        return;
    case Json::value_t::number_float:
        output += python_float_repr(value.get<double>());
        return;
    case Json::value_t::string:
    {
        const std::string& text = value.get_ref<const std::string&>();
        if (text == kNanSentinel)
        {
            output += "NaN";
        }
        else if (text == kPositiveInfinitySentinel)
        {
            output += "Infinity";
        }
        else if (text == kNegativeInfinitySentinel)
        {
            output += "-Infinity";
        }
        else
        {
            append_escaped(output, text);
        }
        return;
    }
    case Json::value_t::array:
    {
        if (value.empty())
        {
            output += "[]";
            return;
        }
        output.push_back('[');
        bool first = true;
        for (const Json& item : value)
        {
            if (!first)
            {
                output.push_back(',');
            }
            if (!compact)
            {
                newline(depth + 1);
            }
            first = false;
            dump_value(output, item, compact, depth + 1);
        }
        if (!compact)
        {
            newline(depth);
        }
        output.push_back(']');
        return;
    }
    case Json::value_t::object:
    {
        if (value.empty())
        {
            output += "{}";
            return;
        }
        output.push_back('{');
        bool first = true;
        for (auto item = value.begin(); item != value.end(); ++item)
        {
            if (!first)
            {
                output.push_back(',');
            }
            if (!compact)
            {
                newline(depth + 1);
            }
            first = false;
            append_escaped(output, item.key());
            output += compact ? ":" : ": ";
            dump_value(output, item.value(), compact, depth + 1);
        }
        if (!compact)
        {
            newline(depth);
        }
        output.push_back('}');
        return;
    }
    default:
        output += "null";
        return;
    }
}

} // namespace

std::string python_float_repr(double value)
{
    if (std::isnan(value))
    {
        return "NaN";
    }
    if (std::isinf(value))
    {
        return value > 0 ? "Infinity" : "-Infinity";
    }
    if (value == 0.0)
    {
        return std::signbit(value) ? "-0.0" : "0.0";
    }
    char buffer[64];
    const std::to_chars_result result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    std::string text(buffer, result.ptr);
    std::string sign;
    if (!text.empty() && text[0] == '-')
    {
        sign = "-";
        text.erase(0, 1);
    }
    const std::size_t exponent_position = text.find('e');
    std::string mantissa = text.substr(0, exponent_position);
    const int exponent = std::stoi(text.substr(exponent_position + 1));
    std::string digits;
    for (char character : mantissa)
    {
        if (character != '.')
        {
            digits.push_back(character);
        }
    }
    while (digits.size() > 1 && digits.back() == '0')
    {
        digits.pop_back();
    }
    const int decimal_point = exponent + 1;
    const int digit_count = static_cast<int>(digits.size());
    std::string output = sign;
    if (decimal_point <= -4 || decimal_point > 16)
    {
        output += digits[0];
        if (digit_count > 1)
        {
            output += '.';
            output += digits.substr(1);
        }
        const int shown = decimal_point - 1;
        char exponent_text[16];
        std::snprintf(exponent_text, sizeof(exponent_text), "e%c%02d", shown < 0 ? '-' : '+', std::abs(shown));
        output += exponent_text;
        return output;
    }
    if (decimal_point <= 0)
    {
        output += "0.";
        output.append(static_cast<std::size_t>(-decimal_point), '0');
        output += digits;
    }
    else if (decimal_point >= digit_count)
    {
        output += digits;
        output.append(static_cast<std::size_t>(decimal_point - digit_count), '0');
        output += ".0";
    }
    else
    {
        output += digits.substr(0, static_cast<std::size_t>(decimal_point));
        output += '.';
        output += digits.substr(static_cast<std::size_t>(decimal_point));
    }
    return output;
}

Json jfloat(double value)
{
    if (std::isnan(value))
    {
        return Json(kNanSentinel);
    }
    if (std::isinf(value))
    {
        return Json(value > 0 ? kPositiveInfinitySentinel : kNegativeInfinitySentinel);
    }
    return Json(value);
}

bool jis_number(const Json& value)
{
    if (value.is_number())
    {
        return true;
    }
    if (value.is_string())
    {
        const std::string& text = value.get_ref<const std::string&>();
        return text == kNanSentinel || text == kPositiveInfinitySentinel || text == kNegativeInfinitySentinel;
    }
    return false;
}

double jnum(const Json& value)
{
    if (value.is_number())
    {
        return value.get<double>();
    }
    if (value.is_string())
    {
        const std::string& text = value.get_ref<const std::string&>();
        if (text == kNanSentinel)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (text == kPositiveInfinitySentinel)
        {
            return std::numeric_limits<double>::infinity();
        }
        if (text == kNegativeInfinitySentinel)
        {
            return -std::numeric_limits<double>::infinity();
        }
    }
    fail_value("JSON value is not a number: " + value.dump());
}

long long jint(const Json& value)
{
    if (value.is_number_integer())
    {
        return value.get<long long>();
    }
    if (value.is_number_unsigned())
    {
        return static_cast<long long>(value.get<unsigned long long>());
    }
    if (value.is_number_float())
    {
        const double number = value.get<double>();
        if (std::floor(number) == number)
        {
            return static_cast<long long>(number);
        }
    }
    if (value.is_boolean())
    {
        return value.get<bool>() ? 1 : 0;
    }
    fail_value("JSON value is not an integer: " + value.dump());
}

const Json& jarray(const Json& object, const char* key)
{
    static const Json empty = Json::array();
    if (!object.is_object())
    {
        return empty;
    }
    const auto found = object.find(key);
    if (found == object.end() || !found->is_array())
    {
        return empty;
    }
    return *found;
}

Json parse_python_json(const std::string& input)
{
    std::string text;
    text.reserve(input.size() + 64);
    std::size_t start = 0;
    if (input.size() >= 3 && static_cast<unsigned char>(input[0]) == 0xEF &&
        static_cast<unsigned char>(input[1]) == 0xBB && static_cast<unsigned char>(input[2]) == 0xBF)
    {
        start = 3;
    }
    bool in_string = false;
    for (std::size_t index = start; index < input.size(); ++index)
    {
        const char character = input[index];
        if (in_string)
        {
            text.push_back(character);
            if (character == '\\' && index + 1 < input.size())
            {
                text.push_back(input[++index]);
            }
            else if (character == '"')
            {
                in_string = false;
            }
            continue;
        }
        if (character == '"')
        {
            in_string = true;
            text.push_back(character);
            continue;
        }
        if (input.compare(index, 3, "NaN") == 0)
        {
            text += "\"\\u0001ai3d-json:NaN\"";
            index += 2;
            continue;
        }
        if (input.compare(index, 8, "Infinity") == 0)
        {
            text += "\"\\u0001ai3d-json:Infinity\"";
            index += 7;
            continue;
        }
        if (input.compare(index, 9, "-Infinity") == 0)
        {
            text += "\"\\u0001ai3d-json:-Infinity\"";
            index += 8;
            continue;
        }
        text.push_back(character);
    }
    try
    {
        return Json::parse(text);
    }
    catch (const std::exception& error)
    {
        throw WaterError(ErrorType::parse, std::string("JSON parse failed: ") + error.what());
    }
}

Json read_json_file(const std::filesystem::path& path)
{
    return parse_python_json(read_file_text(path));
}

std::string dump_python(const Json& value, bool compact)
{
    std::string output;
    dump_value(output, value, compact, 0);
    return output;
}

void write_json_atomic(const std::filesystem::path& path, const Json& value, bool compact)
{
    const std::string text = dump_python(value, compact);
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    write_file_atomic(path, text.data(), text.size(), temporary);
}

} // namespace ai3d::waterpost
