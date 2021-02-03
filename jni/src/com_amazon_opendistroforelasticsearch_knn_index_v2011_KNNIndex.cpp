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

#include "com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex.h"

#include "init.h"
#include "index.h"
#include "java_exception.h"
#include "params.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "spacefactory.h"
#include "space.h"

using std::vector;

using similarity::initLibrary;
using similarity::AnyParams;
using similarity::Index;
using similarity::MethodFactoryRegistry;
using similarity::SpaceFactoryRegistry;
using similarity::AnyParams;
using similarity::Space;
using similarity::ObjectVector;
using similarity::Object;
using similarity::KNNQuery;
using similarity::KNNQueue;

extern "C"

struct IndexWrapper {
  explicit IndexWrapper(const string& spaceType) {
    space.reset(SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceType, AnyParams()));
    index.reset(MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceType, *space, data));
  }
  std::unique_ptr<Space<float>> space;
  std::unique_ptr<Index<float>> index;
  // Index gets constructed with a reference to data (see above) but is otherwise unused
  ObjectVector data;
};

JNIEXPORT void JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_saveIndex(JNIEnv* env, jclass cls, jintArray ids, jobjectArray objectVectors, jstring indexPath, jobjectArray algoParams, jstring spaceType)
{
    Space<float>* space = nullptr;
    ObjectVector dataset;
    Index<float>* index = nullptr;
    int* objectIds = nullptr;

    try {
        const char *spaceTypeCStr = env->GetStringUTFChars(spaceType, nullptr);
        string spaceTypeString(spaceTypeCStr);
        env->ReleaseStringUTFChars(spaceType, spaceTypeCStr);
        has_exception_in_stack(env);
        space = SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceTypeString, AnyParams());
        objectIds = env->GetIntArrayElements(ids, nullptr);

        /*
         * Extract vectors from the Java object array. Efficiently release references when they no longer are needed
         */
        uint objectVectorCount = env->GetArrayLength(objectVectors);
        if (objectVectorCount == 0) {
            return;
        }
        auto objectVectorArray = (jfloatArray)env->GetObjectArrayElement(objectVectors, 0);
        float* objectVector = env->GetFloatArrayElements(objectVectorArray, nullptr);
        uint objectVectorSize = env->GetArrayLength(objectVectorArray)*sizeof(float);
        dataset.push_back(new Object(objectIds[0], -1, objectVectorSize, objectVector));
        env->ReleaseFloatArrayElements(objectVectorArray, objectVector, JNI_ABORT);

        for (uint i = 1; i < objectVectorCount; i++) {
            objectVectorArray = (jfloatArray)env->GetObjectArrayElement(objectVectors, i);
            objectVector = env->GetFloatArrayElements(objectVectorArray, nullptr);
            dataset.push_back(new Object(objectIds[i], -1, objectVectorSize, objectVector));
            env->ReleaseFloatArrayElements(objectVectorArray, objectVector, JNI_ABORT);
        }

        env->DeleteLocalRef(objectVectors);
        env->ReleaseIntArrayElements(ids, objectIds, JNI_ABORT);

        /*
         * Extract parameters from Java array of strings.
         */
        uint paramsCount = env->GetArrayLength(algoParams);
        vector<string> paramsList;
        jstring param;
        const char * rawString;
        for (uint i = 0; i < paramsCount; i++) {
            param = (jstring) (env->GetObjectArrayElement(algoParams, i));
            rawString = env->GetStringUTFChars(param, nullptr);
            paramsList.emplace_back(rawString);
            env->ReleaseStringUTFChars(param, rawString);
        }

        env->DeleteLocalRef(algoParams);

        /*
         * Create Hnsw index for this segment and save it to disk
         */
        index = MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceTypeString, *space, dataset);
        index->CreateIndex(AnyParams(paramsList));
        has_exception_in_stack(env);
        const char *indexString = env->GetStringUTFChars(indexPath, nullptr);
        index->SaveIndex(indexString);
        env->ReleaseStringUTFChars(indexPath, indexString);
        has_exception_in_stack(env);

        // Free each object in the dataset. No need to clear the vector because it goes out of scope
        // immediately
        for (auto & it : dataset) {
             delete it;
        }
        delete index;
        delete space;
    }
    catch (...) {
        if (objectIds) { env->ReleaseIntArrayElements(ids, objectIds, JNI_ABORT); }
        for (auto & it : dataset) {
             delete it;
        }
        if (index) { delete index; }
        if (space) { delete space; }
        catch_cpp_exception_and_throw_java(env);
    }
}

JNIEXPORT jobjectArray JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_queryIndex(JNIEnv* env, jclass cls, jlong indexPointer, jfloatArray queryVector, jint k)
{
    try {
        /*
         * Get pointer to index to be searched
         */
        auto *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointer);

        /*
         * Create the query object that will be used for the search
         */
        float* rawQueryvector = env->GetFloatArrayElements(queryVector, nullptr);
        std::unique_ptr<const Object> queryObject(new Object(-1, -1, env->GetArrayLength(queryVector)*sizeof(float), rawQueryvector));
        env->ReleaseFloatArrayElements(queryVector, rawQueryvector, JNI_ABORT);
        has_exception_in_stack(env);
        KNNQuery<float> knnQuery(*(indexWrapper->space), queryObject.get(), k);

        /*
         * Execute search against index
         */
        indexWrapper->index->Search(&knnQuery);

        /*
         * Copy results to KNNQueryResult
         */
        std::unique_ptr<KNNQueue<float>> result(knnQuery.Result()->Clone());
        has_exception_in_stack(env);
        int resultSize = result->Size();
        jclass resultClass = env->FindClass("com/amazon/opendistroforelasticsearch/knn/index/KNNQueryResult");
        jmethodID allArgs = env->GetMethodID(resultClass, "<init>", "(IF)V");
        jobjectArray results = env->NewObjectArray(resultSize, resultClass, nullptr);

        float distance;
        long id;
        for (uint i = 0; i < resultSize; i++) {
            distance = result->TopDistance();
            id = result->Pop()->id();
            env->SetObjectArrayElement(results, i, env->NewObject(resultClass, allArgs, id, distance));
        }
        has_exception_in_stack(env);
        return results;
    } catch(...) {
        catch_cpp_exception_and_throw_java(env);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_init(JNIEnv* env, jclass cls,  jstring indexPath, jobjectArray algoParams, jstring spaceType)
{
    IndexWrapper *indexWrapper = nullptr;
    try {
        const char *indexPathCStr = env->GetStringUTFChars(indexPath, nullptr);
        string indexPathString(indexPathCStr);
        env->ReleaseStringUTFChars(indexPath, indexPathCStr);
        has_exception_in_stack(env);

        // Load index from file (may throw)
        const char *spaceTypeCStr = env->GetStringUTFChars(spaceType, nullptr);
        string spaceTypeString(spaceTypeCStr);
        env->ReleaseStringUTFChars(spaceType, spaceTypeCStr);
        has_exception_in_stack(env);
        indexWrapper = new IndexWrapper(spaceTypeString);
        indexWrapper->index->LoadIndex(indexPathString);

        // Parse and set query params
        int paramsCount = env->GetArrayLength(algoParams);
        vector<string> paramsList;
        jstring param;
        const char *rawString;
        for (int i=0; i<paramsCount; i++) {
            param = (jstring) (env->GetObjectArrayElement(algoParams, i));
            rawString = env->GetStringUTFChars(param, nullptr);
            paramsList.emplace_back(rawString);
            env->ReleaseStringUTFChars(param, rawString);
        }
        env->DeleteLocalRef(algoParams);
        indexWrapper->index->SetQueryTimeParams(AnyParams(paramsList));
        has_exception_in_stack(env);

        return (jlong) indexWrapper;
    }
    // nmslib seems to throw std::runtime_error if the index cannot be read (which
    // is the only known failure mode for init()).
    catch (...) {
        if (indexWrapper) delete indexWrapper;
        catch_cpp_exception_and_throw_java(env);
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_gc(JNIEnv* env, jclass cls,  jlong indexPointer)
{
    try {
        auto *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointer);
        has_exception_in_stack(env);
        delete indexWrapper;
        has_exception_in_stack(env);
    }
    catch (...) {
        catch_cpp_exception_and_throw_java(env);
    }
}

JNIEXPORT void JNICALL Java_com_amazon_opendistroforelasticsearch_knn_index_v2011_KNNIndex_initLibrary(JNIEnv *, jclass)
{
    initLibrary();
}
