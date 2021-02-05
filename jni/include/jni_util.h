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

#ifndef OPENDISTRO_KNN_JNI_UTIL_H
#define OPENDISTRO_KNN_JNI_UTIL_H

#include <jni.h>
#include <stdexcept>
#include <new>

void java_exception(JNIEnv* env, const char* type = "", const char* message = "") {
    jclass newExcCls = env->FindClass(type);
    if (newExcCls != nullptr) {
        env->ThrowNew(newExcCls, message);
    }
    //If newExcCls isn't found, NoClassDefFoundError will be thrown
}

inline void has_exception_in_stack(JNIEnv* env)
{
    if (env->ExceptionCheck() == JNI_TRUE) {
        throw std::runtime_error("Exception Occurred");
    }
}

void catch_cpp_exception_and_throw_java(JNIEnv* env)
{
    try {
        throw;
    }
    catch (const std::bad_alloc& rhs) {
        java_exception(env, "java/io/IOException", rhs.what());
    }
    catch (const std::runtime_error& re) {
        java_exception(env, "java/lang/Exception", re.what());
    }
    catch (const std::exception& e) {
        java_exception(env, "java/lang/Exception", e.what());
    }
    catch (...) {
        java_exception(env, "java/lang/Exception", "Unknown exception occurred");
    }
}

#endif //OPENDISTRO_KNN_JNI_UTIL_H