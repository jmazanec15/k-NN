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

import org.elasticsearch.common.ValidationException;

import java.util.function.Predicate;

/**
 * Parameter that can be set for a method component
 *
 * @param <T> Type parameter takes
 */
public abstract class Parameter<T> {

    private T defaultValue;
    private boolean inMethodString;
    protected Predicate<T> validator;

    /**
     * Constructor
     *
     * @param defaultValue of the parameter
     * @param inMethodString whether the parameter is included in method string
     * @param validator used to validate a parameter value passed
     */
    public Parameter(T defaultValue, boolean inMethodString, Predicate<T> validator) {
        this.defaultValue = defaultValue;
        this.inMethodString = inMethodString;
        this.validator = validator;
    }

    /**
     * Get default value for parameter
     *
     * @return default value of the parameter
     */
    public T getDefaultValue() {
        return defaultValue;
    }

    /**
     * Is the parameter included in the method string
     *
     * @return true if the parameter should be included in the method string; false otherwise
     */
    public boolean isInMethodString() {
        return inMethodString;
    }

    /**
     * Check if the value passed in is valid
     *
     * @param value to be checked
     */
    public abstract void validate(Object value);

    /**
     * Integer method parameter
     */
    public static class IntegerParameter extends Parameter<Integer> {
        public IntegerParameter(Integer defaultValue, boolean inMethodString, Predicate<Integer> validator)
        {
            super(defaultValue, inMethodString, validator);
        }

        @Override
        public void validate(Object value) {
            if (!(value instanceof Integer) || !validator.test((Integer) value)) {
                throw new ValidationException();
            }
        }
    }
}
