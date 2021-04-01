/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.opendistroforelasticsearch.knn.index;

import com.amazon.opendistroforelasticsearch.knn.index.util.KNNEngine;

import java.util.Map;
import java.util.Set;

public class KNNMethod {

    public KNNMethod(String name, Set<SpaceTypes> validSpaces, Map<String, Class<?>> validParameters,
                     Map<String, KNNEncoder> validEncoders, boolean isCourseQuantizerAvailable) {
        this.name = name;
        this.validSpaces = validSpaces;
        this.validParameters = validParameters;
        this.validEncoders = validEncoders;
        this.isCourseQuantizerAvailable = isCourseQuantizerAvailable;
    }

    private String name;
    private Set<SpaceTypes> validSpaces;
    private Map<String, Class<?>> validParameters;
    private Map<String, KNNEncoder> validEncoders;
    private boolean isCourseQuantizerAvailable;

    public boolean validate(KNNMethodContext knnMethodContext, KNNEngine knnEngine) {
        //TODO: Probably need to throw exception instead of boolean

        // Validate parameters
        if (knnMethodContext.getParameters() != null) {
            for (Map.Entry<String, Object> parameter : knnMethodContext.getParameters().entrySet()) {
                if (!validParameters.containsKey(parameter.getKey())) {
                    return false;
                }

                if (validParameters.get(parameter.getKey()) != parameter.getValue().getClass()) {
                    return false;
                }
            }
        }


        // Validate CourseQuantizer
        KNNMethodContext courseQuantizer = knnMethodContext.getCourseQuantizer();
        if (courseQuantizer != null) {
            if (!isCourseQuantizerAvailable || !knnEngine.validateMethod(courseQuantizer)) {
                return false;
            }
        }

        // Validate Encoder
        KNNMethodContext.ComponentContext encoderContext = knnMethodContext.getEncoding();

        if (encoderContext != null) {
            if (!validEncoders.containsKey(encoderContext.getName())) {
                return false;
            }

            if (!validEncoders.get(encoderContext.getName()).validate(encoderContext)) {
                return false;
            }
        }
        return true;
    }

    public KNNEncoder getEncoder(String encoderName) {
        if (!validEncoders.containsKey(encoderName)) {
            throw new IllegalArgumentException("Invalid encoder: " + encoderName);
        }
        return validEncoders.get(encoderName);
    }

    public String getName() {
        return name;
    }
}
