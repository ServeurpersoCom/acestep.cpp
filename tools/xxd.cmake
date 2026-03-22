# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
# Converts any binary file into a C array. Pure CMake, no external tools.
# Works on Linux, macOS, and Windows (no xxd, no bash required).
# Usage: cmake -DINPUT=path/to/file -DOUTPUT=path/to/file.hpp -P xxd.cmake

file(READ "${INPUT}" hex_data HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_sequence "${hex_data}")

string(LENGTH ${hex_data} hex_len)
math(EXPR len "${hex_len} / 2")

get_filename_component(filename "${INPUT}" NAME)
string(REGEX REPLACE "\\.|-" "_" name "${filename}")

file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {${hex_sequence}};\nunsigned int ${name}_len = ${len};\n")
