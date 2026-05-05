#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>

namespace gpt {

enum class LogLevel { DEBUG = 0, INFO, WARN, ERROR };

class Logger {
public:
    static Logger& get() { static Logger instance; return instance; }

    void set_level(LogLevel lvl)             { level_ = lvl; }
    void set_file(const std::string& path)   { file_.open(path, std::ios::app); }

    template<typename... Args>
    void log(LogLevel lvl, Args&&... args) {
        if (lvl < level_) return;
        std::ostringstream oss;
        oss << "[" << timestamp() << "] [" << level_str(lvl) << "] ";
        (oss << ... << std::forward<Args>(args));
        oss << "\n";
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << oss.str();
        if (file_.is_open()) file_ << oss.str();
    }

    template<typename... Args> void debug(Args&&... a) { log(LogLevel::DEBUG, std::forward<Args>(a)...); }
    template<typename... Args> void info (Args&&... a) { log(LogLevel::INFO,  std::forward<Args>(a)...); }
    template<typename... Args> void warn (Args&&... a) { log(LogLevel::WARN,  std::forward<Args>(a)...); }
    template<typename... Args> void error(Args&&... a) { log(LogLevel::ERROR, std::forward<Args>(a)...); }

private:
    Logger() = default;
    LogLevel          level_{LogLevel::INFO};
    std::ofstream     file_;
    std::mutex        mtx_;

    static std::string timestamp() {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time), "%H:%M:%S");
        return oss.str();
    }

    static const char* level_str(LogLevel l) {
        switch (l) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
        }
        return "?????";
    }
};

// Convenience macros
#define LOG_DEBUG(...) gpt::Logger::get().debug(__VA_ARGS__)
#define LOG_INFO(...)  gpt::Logger::get().info(__VA_ARGS__)
#define LOG_WARN(...)  gpt::Logger::get().warn(__VA_ARGS__)
#define LOG_ERROR(...) gpt::Logger::get().error(__VA_ARGS__)

} // namespace gpt
