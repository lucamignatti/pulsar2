cmake_minimum_required(VERSION 3.20)
project(CdoDumpPlugin LANGUAGES CXX)

if(NOT WIN32)
	message(FATAL_ERROR "Bakkes plugins are Windows-only.")
endif()

set(BAKKES "$ENV{APPDATA}/bakkesmod/bakkesmod")
set(SDK "${BAKKES}/bakkesmodsdk")
if(NOT EXISTS "${SDK}/include/bakkesmod/plugin/bakkesmodplugin.h")
	message(FATAL_ERROR "Bakkes SDK not found at ${SDK}")
endif()

add_library(CdoDumpPlugin SHARED CdoDumpPlugin.cpp CdoDumpPlugin.h)
target_include_directories(CdoDumpPlugin PRIVATE "${SDK}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(CdoDumpPlugin PRIVATE "${SDK}/lib/pluginsdk.lib")
target_compile_features(CdoDumpPlugin PRIVATE cxx_std_17)
target_compile_definitions(CdoDumpPlugin PRIVATE
	_CRT_SECURE_NO_WARNINGS
	WIN32_LEAN_AND_MEAN
	NOMINMAX
)
if(MSVC)
	target_compile_options(CdoDumpPlugin PRIVATE /W3 /EHsc /MP /permissive-)
endif()

add_custom_command(TARGET CdoDumpPlugin POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory "${BAKKES}/plugins"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:CdoDumpPlugin>
		"${BAKKES}/plugins/CdoDumpPlugin.dll"
	COMMENT "Install CdoDumpPlugin.dll into Bakkes plugins/"
)
