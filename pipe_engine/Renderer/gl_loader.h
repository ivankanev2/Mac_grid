#pragma once

// Centralized OpenGL include / loader glue.
//
// macOS keeps using the system OpenGL 3 header exactly as before.
// Linux / other Unix-like builds keep using GL_GLEXT_PROTOTYPES so the
// existing direct-call code path continues to work.
//
// Windows is the special case: opengl32 only exports OpenGL 1.1 symbols, so
// modern entry points (VBOs, VAOs, shaders, 3D textures, etc.) must be loaded
// from the active context via glfwGetProcAddress(). We keep that logic local
// to this header so the rest of the code can continue calling glFoo(...)
// unchanged on every platform.

#ifdef __APPLE__
#  define GL_SILENCE_DEPRECATION
#  include <OpenGL/gl3.h>

namespace pipe_gl {
inline bool ensureLoaded() { return true; }
inline const char* missingProcName() { return nullptr; }
} // namespace pipe_gl

#elif defined(_WIN32)

#  include <GL/gl.h>
#  include <GL/glext.h>

extern "C" {
typedef void (*GLFWglproc)(void);
GLFWglproc glfwGetProcAddress(const char* procname);
}

namespace pipe_gl {

inline const char* g_missingProc = nullptr;
inline bool g_loaded = false;

template <typename T>
inline bool loadProc(T& dst, const char* name) {
    dst = reinterpret_cast<T>(glfwGetProcAddress(name));
    if (!dst) {
        if (!g_missingProc) g_missingProc = name;
        return false;
    }
    return true;
}

inline PFNGLACTIVETEXTUREPROC           fn_glActiveTexture           = nullptr;
inline PFNGLATTACHSHADERPROC            fn_glAttachShader            = nullptr;
inline PFNGLBINDATTRIBLOCATIONPROC      fn_glBindAttribLocation      = nullptr;
inline PFNGLBINDBUFFERPROC              fn_glBindBuffer              = nullptr;
inline PFNGLBINDVERTEXARRAYPROC         fn_glBindVertexArray         = nullptr;
inline PFNGLBLENDFUNCSEPARATEPROC       fn_glBlendFuncSeparate       = nullptr;
inline PFNGLBUFFERDATAPROC              fn_glBufferData              = nullptr;
inline PFNGLCOMPILESHADERPROC           fn_glCompileShader           = nullptr;
inline PFNGLCREATEPROGRAMPROC           fn_glCreateProgram           = nullptr;
inline PFNGLCREATESHADERPROC            fn_glCreateShader            = nullptr;
inline PFNGLDELETEBUFFERSPROC           fn_glDeleteBuffers           = nullptr;
inline PFNGLDELETEPROGRAMPROC           fn_glDeleteProgram           = nullptr;
inline PFNGLDELETESHADERPROC            fn_glDeleteShader            = nullptr;
inline PFNGLDELETEVERTEXARRAYSPROC      fn_glDeleteVertexArrays      = nullptr;
inline PFNGLENABLEVERTEXATTRIBARRAYPROC fn_glEnableVertexAttribArray = nullptr;
inline PFNGLGENBUFFERSPROC              fn_glGenBuffers              = nullptr;
inline PFNGLGENVERTEXARRAYSPROC         fn_glGenVertexArrays         = nullptr;
inline PFNGLGETPROGRAMINFOLOGPROC       fn_glGetProgramInfoLog       = nullptr;
inline PFNGLGETPROGRAMIVPROC            fn_glGetProgramiv            = nullptr;
inline PFNGLGETSHADERINFOLOGPROC        fn_glGetShaderInfoLog        = nullptr;
inline PFNGLGETSHADERIVPROC             fn_glGetShaderiv             = nullptr;
inline PFNGLGETUNIFORMLOCATIONPROC      fn_glGetUniformLocation      = nullptr;
inline PFNGLLINKPROGRAMPROC             fn_glLinkProgram             = nullptr;
inline PFNGLSHADERSOURCEPROC            fn_glShaderSource            = nullptr;
inline PFNGLTEXIMAGE3DPROC              fn_glTexImage3D              = nullptr;
inline PFNGLTEXSUBIMAGE3DPROC           fn_glTexSubImage3D           = nullptr;
inline PFNGLUNIFORM1FPROC               fn_glUniform1f               = nullptr;
inline PFNGLUNIFORM1IPROC               fn_glUniform1i               = nullptr;
inline PFNGLUNIFORM3FPROC               fn_glUniform3f               = nullptr;
inline PFNGLUNIFORM3FVPROC              fn_glUniform3fv              = nullptr;
inline PFNGLUNIFORMMATRIX3FVPROC        fn_glUniformMatrix3fv        = nullptr;
inline PFNGLUNIFORMMATRIX4FVPROC        fn_glUniformMatrix4fv        = nullptr;
inline PFNGLUSEPROGRAMPROC              fn_glUseProgram              = nullptr;
inline PFNGLVERTEXATTRIBPOINTERPROC     fn_glVertexAttribPointer     = nullptr;

inline bool ensureLoaded() {
    if (g_loaded) return true;

    g_missingProc = nullptr;
    bool ok = true;

    ok = loadProc(fn_glActiveTexture,           "glActiveTexture")           && ok;
    ok = loadProc(fn_glAttachShader,            "glAttachShader")            && ok;
    ok = loadProc(fn_glBindAttribLocation,      "glBindAttribLocation")      && ok;
    ok = loadProc(fn_glBindBuffer,              "glBindBuffer")              && ok;
    ok = loadProc(fn_glBindVertexArray,         "glBindVertexArray")         && ok;
    ok = loadProc(fn_glBlendFuncSeparate,       "glBlendFuncSeparate")       && ok;
    ok = loadProc(fn_glBufferData,              "glBufferData")              && ok;
    ok = loadProc(fn_glCompileShader,           "glCompileShader")           && ok;
    ok = loadProc(fn_glCreateProgram,           "glCreateProgram")           && ok;
    ok = loadProc(fn_glCreateShader,            "glCreateShader")            && ok;
    ok = loadProc(fn_glDeleteBuffers,           "glDeleteBuffers")           && ok;
    ok = loadProc(fn_glDeleteProgram,           "glDeleteProgram")           && ok;
    ok = loadProc(fn_glDeleteShader,            "glDeleteShader")            && ok;
    ok = loadProc(fn_glDeleteVertexArrays,      "glDeleteVertexArrays")      && ok;
    ok = loadProc(fn_glEnableVertexAttribArray, "glEnableVertexAttribArray") && ok;
    ok = loadProc(fn_glGenBuffers,              "glGenBuffers")              && ok;
    ok = loadProc(fn_glGenVertexArrays,         "glGenVertexArrays")         && ok;
    ok = loadProc(fn_glGetProgramInfoLog,       "glGetProgramInfoLog")       && ok;
    ok = loadProc(fn_glGetProgramiv,            "glGetProgramiv")            && ok;
    ok = loadProc(fn_glGetShaderInfoLog,        "glGetShaderInfoLog")        && ok;
    ok = loadProc(fn_glGetShaderiv,             "glGetShaderiv")             && ok;
    ok = loadProc(fn_glGetUniformLocation,      "glGetUniformLocation")      && ok;
    ok = loadProc(fn_glLinkProgram,             "glLinkProgram")             && ok;
    ok = loadProc(fn_glShaderSource,            "glShaderSource")            && ok;
    ok = loadProc(fn_glTexImage3D,              "glTexImage3D")              && ok;
    ok = loadProc(fn_glTexSubImage3D,           "glTexSubImage3D")           && ok;
    ok = loadProc(fn_glUniform1f,               "glUniform1f")               && ok;
    ok = loadProc(fn_glUniform1i,               "glUniform1i")               && ok;
    ok = loadProc(fn_glUniform3f,               "glUniform3f")               && ok;
    ok = loadProc(fn_glUniform3fv,              "glUniform3fv")              && ok;
    ok = loadProc(fn_glUniformMatrix3fv,        "glUniformMatrix3fv")        && ok;
    ok = loadProc(fn_glUniformMatrix4fv,        "glUniformMatrix4fv")        && ok;
    ok = loadProc(fn_glUseProgram,              "glUseProgram")              && ok;
    ok = loadProc(fn_glVertexAttribPointer,     "glVertexAttribPointer")     && ok;

    g_loaded = ok;
    return ok;
}

inline const char* missingProcName() { return g_missingProc; }

} // namespace pipe_gl

#  define glActiveTexture           pipe_gl::fn_glActiveTexture
#  define glAttachShader            pipe_gl::fn_glAttachShader
#  define glBindAttribLocation      pipe_gl::fn_glBindAttribLocation
#  define glBindBuffer              pipe_gl::fn_glBindBuffer
#  define glBindVertexArray         pipe_gl::fn_glBindVertexArray
#  define glBlendFuncSeparate       pipe_gl::fn_glBlendFuncSeparate
#  define glBufferData              pipe_gl::fn_glBufferData
#  define glCompileShader           pipe_gl::fn_glCompileShader
#  define glCreateProgram           pipe_gl::fn_glCreateProgram
#  define glCreateShader            pipe_gl::fn_glCreateShader
#  define glDeleteBuffers           pipe_gl::fn_glDeleteBuffers
#  define glDeleteProgram           pipe_gl::fn_glDeleteProgram
#  define glDeleteShader            pipe_gl::fn_glDeleteShader
#  define glDeleteVertexArrays      pipe_gl::fn_glDeleteVertexArrays
#  define glEnableVertexAttribArray pipe_gl::fn_glEnableVertexAttribArray
#  define glGenBuffers              pipe_gl::fn_glGenBuffers
#  define glGenVertexArrays         pipe_gl::fn_glGenVertexArrays
#  define glGetProgramInfoLog       pipe_gl::fn_glGetProgramInfoLog
#  define glGetProgramiv            pipe_gl::fn_glGetProgramiv
#  define glGetShaderInfoLog        pipe_gl::fn_glGetShaderInfoLog
#  define glGetShaderiv             pipe_gl::fn_glGetShaderiv
#  define glGetUniformLocation      pipe_gl::fn_glGetUniformLocation
#  define glLinkProgram             pipe_gl::fn_glLinkProgram
#  define glShaderSource            pipe_gl::fn_glShaderSource
#  define glTexImage3D              pipe_gl::fn_glTexImage3D
#  define glTexSubImage3D           pipe_gl::fn_glTexSubImage3D
#  define glUniform1f               pipe_gl::fn_glUniform1f
#  define glUniform1i               pipe_gl::fn_glUniform1i
#  define glUniform3f               pipe_gl::fn_glUniform3f
#  define glUniform3fv              pipe_gl::fn_glUniform3fv
#  define glUniformMatrix3fv        pipe_gl::fn_glUniformMatrix3fv
#  define glUniformMatrix4fv        pipe_gl::fn_glUniformMatrix4fv
#  define glUseProgram              pipe_gl::fn_glUseProgram
#  define glVertexAttribPointer     pipe_gl::fn_glVertexAttribPointer

#else

#  ifndef GL_GLEXT_PROTOTYPES
#    define GL_GLEXT_PROTOTYPES
#  endif
#  include <GL/gl.h>
#  include <GL/glext.h>

namespace pipe_gl {
inline bool ensureLoaded() { return true; }
inline const char* missingProcName() { return nullptr; }
} // namespace pipe_gl

#endif
