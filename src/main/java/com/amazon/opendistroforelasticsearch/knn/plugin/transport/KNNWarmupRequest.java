/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.opendistroforelasticsearch.knn.plugin.transport;

import org.elasticsearch.action.support.nodes.BaseNodesRequest;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Transport request to get nodes to load indices into cache
 */
public class KNNWarmupRequest extends BaseNodesRequest<KNNWarmupRequest> {
    private String[] indices;

    public KNNWarmupRequest(StreamInput in) throws IOException {
        super(in);
        indices = in.readStringArray();
    }

    public KNNWarmupRequest(String[] indices, String... nodeIds) {
        super(nodeIds);
        this.indices = indices;
    }

    public String[] getIndices() {
        return this.indices;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeStringArray(indices);
    }
}