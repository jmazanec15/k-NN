/*
 *   Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

#include <jni.h>
#include <stdexcept>
#include <new>

/*
 * This function basically checks if there is an exception in t
 */
inline void has_exception_in_stack(JNIEnv* env)
{
    if (env->ExceptionCheck() == JNI_TRUE) {
        throw std::runtime_error("Exception Occurred");
    }
}

void throw_java_exception(JNIEnv* env, const char* type = "", const char* message = "") {
    env->ExceptionClear();
    jclass newExcCls = env->FindClass(type);
    if (newExcCls != nullptr) {
        env->ThrowNew(newExcCls, message);
    }
}

void catch_cpp_exception_and_throw_java(JNIEnv* env)
{
    try {
        throw;
    }
    catch (const std::bad_alloc& rhs) {
        throw_java_exception(env, "java/io/IOException", rhs.what());
    }
    catch (const std::runtime_error& re) {
        throw_java_exception(env, "java/lang/Exception", re.what());
    }
    catch (const std::exception& e) {
        throw_java_exception(env, "java/lang/Exception", e.what());
    }
    catch (...) {
        throw_java_exception(env, "java/lang/Exception", "Unknown exception occurred");
    }
}
