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

package com.amazon.opendistroforelasticsearch.knn.index.util;

public enum FAISSLibVersion {

    /**
     * Latest available faiss version
     */
    VFAISS_165("FAISS_165") {
        @Override
        public String indexLibraryVersion() {
            return "KNNIndex_FAISS_V1_6_5";
        }
    };

    public static final FAISSLibVersion LATEST = VFAISS_165;

    private String buildVersion;

    FAISSLibVersion(String buildVersion) {
        this.buildVersion = buildVersion;
    }

    /**
     * FAISS library version used by the KNN codec
     * @return name
     */
    public abstract String indexLibraryVersion();

    public String getBuildVersion() { return buildVersion; }
}
