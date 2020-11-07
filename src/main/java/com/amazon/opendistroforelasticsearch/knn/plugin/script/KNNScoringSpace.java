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

import com.amazon.opendistroforelasticsearch.knn.index.KNNVectorFieldMapper;
import com.amazon.opendistroforelasticsearch.knn.plugin.stats.KNNCounter;
import org.apache.lucene.index.LeafReaderContext;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.NumberFieldMapper;
import org.elasticsearch.script.ScoreScript;
import org.elasticsearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import static org.elasticsearch.index.mapper.NumberFieldMapper.NumberType.LONG;

public abstract class KNNScoringSpace {

    protected Object processedQuery;
    protected BiFunction<?, ?, Float> scoringMethod;
    protected MappedFieldType fieldType;

    public KNNScoringSpace(Object query, MappedFieldType fieldType) {
        this.fieldType = fieldType;
        prepareQuery(query);
    }

    /**
     * Prepare the query and the scoring method for the given FieldType and similarity space. When preparing these
     * properties, the fieldType and the spaceType will dictate if they are validated and how they are processed. This
     * method has the responsibility of ensuring that the query and scoring method that will be used when scoring the
     * docs in an index is compatible.
     *
     * @param query Raw query object passed in to be validated and processed for the given similarity space
     */
    public abstract void prepareQuery(Object query);

    /**
     * Return the correct scoring script for a given query. The scoring script
     *
     * @param params Map of parameters
     * @param field Fieldname
     * @param lookup SearchLookup
     * @param ctx ctx LeafReaderContext to be used for scoring documents
     * @return ScoreScript for this query
     * @throws IOException throws IOException if ScoreScript cannot be constructed
     */
    public abstract ScoreScript getScoreScript(Map<String, Object> params, String field, SearchLookup lookup,
                                               LeafReaderContext ctx) throws IOException;

    public static class HammingBitSpace extends KNNScoringSpace {
        public HammingBitSpace(Object query, MappedFieldType fieldType) {
            super(query, fieldType);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void prepareQuery(Object query) {
            if (!(fieldType instanceof NumberFieldMapper.NumberFieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for hamming space. The field type must " +
                        "be an integral numeric type.");
            }

            if (((NumberFieldMapper.NumberFieldType) fieldType).numericType() == LONG.numericType()) {
                /*
                 * Because an Elasticsearch field can have 0 or more values, the query should be processed as a list
                 * of Longs. Additionally, because there is no way to specify the type of integral that is passed in
                 * during query, it is necessary to cast it to a long, or a list of longs here.
                 */
                List<Long> processedQueryList;
                if (query instanceof Integer) {
                    processedQueryList = Collections.singletonList(Long.valueOf((Integer) query));
                } else if (query instanceof Long) {
                    processedQueryList = Collections.singletonList((Long) query);
                } else if (query instanceof List && ((List<?>) query).iterator().next() instanceof Integer) {
                    /*
                     * Need to reverse the list because Elasticsearch stores lists in reverse order. Because this
                     * happens once per query, this does not incur a major latency penalty
                     */
                    processedQueryList = ((List<Integer>) query).stream().mapToLong(Integer::longValue).boxed()
                            .collect(Collectors.toList());
                    Collections.reverse(processedQueryList);
                } else if (query instanceof List && ((List<?>) query).iterator().next() instanceof Long) {
                    processedQueryList = (List<Long>) query;
                    Collections.reverse(processedQueryList);
                } else if (!(query instanceof List) || (((List<?>) query).size() != 0 &&
                                !(((List<?>) query).iterator().next() instanceof Long))
                ) {
                    throw new IllegalArgumentException("Incompatible query_value for hamming space. query_value must " +
                            "be either a Long, an Integer, an array of Longs, or an array of Integers.");
                } else {
                    processedQueryList = Collections.emptyList();
                }

                this.processedQuery = processedQueryList;
                this.scoringMethod = (List<Long> q, List<Long> v) -> 1.0f/(1 + KNNScoringUtil.bitHamming(q, v));
            } else {
                throw new IllegalArgumentException("Incompatible field_type for hamming space. The field type must " +
                        "of type Long.");
            }
        }

        @Override
        @SuppressWarnings("unchecked")
        public ScoreScript getScoreScript(Map<String, Object> params, String field, SearchLookup lookup,
                                                   LeafReaderContext ctx) throws IOException {
            if (((NumberFieldMapper.NumberFieldType) fieldType).numericType() == LONG.numericType()) {
                return new KNNScoreScript.KNNLongListScoreScript(params, (List<Long>) this.processedQuery, field,
                        (BiFunction<List<Long>, List<Long>, Float>) this.scoringMethod, lookup, ctx);
            } else {
                throw new IllegalArgumentException("Incompatible field_type for hamming space. The field type must " +
                        "of type Long.");
            }
        }
    }

    public static class L2Space extends KNNScoringSpace {

        public L2Space(Object query, MappedFieldType fieldType) {
            super(query, fieldType);
        }

        @Override
        public void prepareQuery(Object query) {
            if (!(fieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for l2 space. The field type must " +
                        "be a knn_vector.");
            }

            this.processedQuery = KNNScoringUtil.convertVectorToPrimitive(query);
            if (((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension() !=
                    ((float[]) processedQuery).length) {
                KNNCounter.SCRIPT_QUERY_ERRORS.increment();
                throw new IllegalStateException("[KNN] query vector and field vector dimensions mismatch. " +
                        "query vector: " + ((float[]) processedQuery).length + ", stored vector: " +
                        ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension());
            }
            this.scoringMethod = (float[] q, float[] v) -> 1 / (1 + KNNScoringUtil.l2Squared(q, v));
        }

        @Override
        @SuppressWarnings("unchecked")
        public ScoreScript getScoreScript(Map<String, Object> params, String field, SearchLookup lookup,
                                          LeafReaderContext ctx) throws IOException {
            return new KNNScoreScript.KNNVectorScoreScript(params, (float[]) processedQuery, field,
                    (BiFunction<float[], float[], Float>) this.scoringMethod, lookup, ctx);

        }
    }

    public static class CosineSimilaritySpace extends KNNScoringSpace {

        public CosineSimilaritySpace(Object query, MappedFieldType fieldType) {
            super(query, fieldType);
        }

        @Override
        public void prepareQuery(Object query) {
            if (!(fieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
                throw new IllegalArgumentException("Incompatible field_type for cosine space. The field type must " +
                        "be a knn_vector.");
            }

            this.processedQuery = KNNScoringUtil.convertVectorToPrimitive(query);
            if (((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension() !=
                    ((float[]) processedQuery).length) {
                KNNCounter.SCRIPT_QUERY_ERRORS.increment();
                throw new IllegalStateException("[KNN] query vector and field vector dimensions mismatch. " +
                        "query vector: " + ((float[]) processedQuery).length + ", stored vector: " +
                        ((KNNVectorFieldMapper.KNNVectorFieldType) fieldType).getDimension());
            }
            float qVectorSquaredMagnitude = KNNScoringUtil.getVectorMagnitudeSquared((float[]) this.processedQuery);
            this.scoringMethod = (float[] q, float[] v) -> 1 + KNNScoringUtil.cosinesimilOptimized(q, v,
                    qVectorSquaredMagnitude);
        }

        @Override
        @SuppressWarnings("unchecked")
        public ScoreScript getScoreScript(Map<String, Object> params, String field, SearchLookup lookup,
                                          LeafReaderContext ctx) throws IOException {
                return new KNNScoreScript.KNNVectorScoreScript(params, (float[]) processedQuery, field,
                        (BiFunction<float[], float[], Float>) this.scoringMethod, lookup, ctx);
        }
    }
}
