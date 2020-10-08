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

package com.amazon.opendistroforelasticsearch.knn.plugin.script;

import com.amazon.opendistroforelasticsearch.knn.index.util.KNNConstants;
import com.amazon.opendistroforelasticsearch.knn.plugin.stats.KNNCounter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.script.ScoreScript;
import org.elasticsearch.search.lookup.SearchLookup;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.UncheckedIOException;
import java.util.Map;

/**
 * Vector score script used for adjusting the score based on similarity space
 * on a per document basis.
 *
 */
public class KNNVectorScoreScript extends ScoreScript {

    private BinaryDocValues binaryDocValuesReader;
    private final float[] queryVector;
    private final String similaritySpace;
    private float queryVectorSquaredMagnitude = -1;
    private boolean vectorExist = true;

    /**
     * This function called for each doc in the segment. We evaluate the score of the vector in the doc
     *
     * @param explanationHolder A helper to take in an explanation from a script and turn
     *                          it into an {@link org.apache.lucene.search.Explanation}
     * @return score of the vector to the query vector
     */
    @Override
    public double execute(ScoreScript.ExplanationHolder explanationHolder) {
        // If this document does not contain the vector, push it to end of the results.
        if (!vectorExist) {
            return Float.MIN_VALUE;
        }

        float score = Float.MIN_VALUE;
        try {
            float[] doc_vector;
            BytesRef bytesref = binaryDocValuesReader.binaryValue();
            // If there is no vector for the corresponding doc then it should not be considered for nearest
            // neighbors.
            if (bytesref == null) {
                return Float.MIN_VALUE;
            }
            try (ByteArrayInputStream byteStream = new ByteArrayInputStream(bytesref.bytes, bytesref.offset, bytesref.length);
                 ObjectInputStream objectStream = new ObjectInputStream(byteStream)) {
                doc_vector = (float[]) objectStream.readObject();
            } catch (ClassNotFoundException e) {
                KNNCounter.SCRIPT_QUERY_ERRORS.increment();
                throw new RuntimeException(e);
            }

            if(doc_vector.length != queryVector.length) {
                KNNCounter.SCRIPT_QUERY_ERRORS.increment();
                throw new IllegalStateException("[KNN] query vector and field vector dimensions mismatch. " +
                        "query vector: " + queryVector.length + ", stored vector: " + doc_vector.length);
            }

            if (KNNConstants.L2.equalsIgnoreCase(similaritySpace)) {
                score = KNNScoringUtil.l2Squared(this.queryVector, doc_vector);
                score = 1/(1 + score);
            } else if (KNNConstants.COSINESIMIL.equalsIgnoreCase(similaritySpace)) {
                // Scores cannot be negative so add +1 to the cosine score
                score = 1 + KNNScoringUtil.cosinesimilOptimized(this.queryVector, doc_vector, this.queryVectorSquaredMagnitude);
            }
        } catch (IOException e) {
            KNNCounter.SCRIPT_QUERY_ERRORS.increment();
            throw new UncheckedIOException(e);
        }
        return score;
    }

    @Override
    public void setDocument(int docId) {
        try {
            this.vectorExist = this.binaryDocValuesReader.advanceExact(docId);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public KNNVectorScoreScript(Map<String, Object> params, String field, float[] queryVector, float queryVectorSquaredMagnitude,
                                String similaritySpace, SearchLookup lookup, LeafReaderContext leafContext) throws IOException {
        super(params, lookup, leafContext);
        // get query vector - convert to primitive
        final Object vector = params.get("vector");
        this.similaritySpace = similaritySpace;
        this.queryVector = queryVector;
        this.queryVectorSquaredMagnitude = queryVectorSquaredMagnitude;
        this.binaryDocValuesReader = leafContext.reader().getBinaryDocValues(field);
        if(this.binaryDocValuesReader == null) {
            KNNCounter.SCRIPT_QUERY_ERRORS.increment();
            throw new IllegalStateException("Binary Doc values not enabled for the field " + field
                                                        + " Please ensure the field type is knn_vector in mappings for this field");
        }
    }
}
