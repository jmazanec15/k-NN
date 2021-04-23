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

package com.amazon.opendistroforelasticsearch.knn.index;

import com.amazon.opendistroforelasticsearch.knn.KNNTestCase;
import com.amazon.opendistroforelasticsearch.knn.index.faiss.v165.KNNFaissIndex;
import com.amazon.opendistroforelasticsearch.knn.index.nmslib.v2011.KNNNmsLibIndex;
import com.amazon.opendistroforelasticsearch.knn.index.util.KNNEngine;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;

import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Collectors;

import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.METHOD_HNSW;
import static com.amazon.opendistroforelasticsearch.knn.index.KNNMethodContext.MIN_MINIMUM_DATAPOINTS;
import static com.amazon.opendistroforelasticsearch.knn.index.KNNMethodContext.MIN_TRAINING_DATASET_SIZE_LIMIT;

public class KNNJNITests extends KNNTestCase {

    private static final Logger logger = LogManager.getLogger(KNNJNINmsLibTests.class);

    private int[] docs = {0, 1, 2};

    private float[][] vectors = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f}
    };

    public void testCreateMixedEngineIndex() throws Exception {
        Directory dir = newFSDirectory(createTempDir());
        String segmentName = "_dummy";
        {
            String indexPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                    String.format("%s_%s%s", segmentName, KNNEngine.NMSLIB.getName(),
                            KNNEngine.NMSLIB.getExtension())).toString();
            String[] algoParams = {};
            AccessController.doPrivileged(
                    (PrivilegedAction<Void>) () -> {
                        KNNNmsLibIndex.save(docs, vectors, indexPath, algoParams, SpaceType.L2.getValue(),
                                METHOD_HNSW);
                        return null;
                    }
            );
        }
        {
            String indexPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                    String.format("%s_%s%s", segmentName, KNNEngine.FAISS.getName(),
                            KNNEngine.FAISS.getExtension())).toString();
            String[] algoParams = {};



            AccessController.doPrivileged(
                    (PrivilegedAction<Void>) () -> {
                        KNNFaissIndex.save(docs, vectors, indexPath, Collections.emptyMap(),
                                SpaceType.L2.getValue(), "Flat",
                                MIN_TRAINING_DATASET_SIZE_LIMIT + 1, MIN_MINIMUM_DATAPOINTS + 1);
                        return null;
                    }
            );
        }
        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy_NMSLIB" + KNNEngine.NMSLIB.getExtension()));
        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy_FAISS" + KNNEngine.FAISS.getExtension()));
        dir.close();
    }

    public void testQueryMixedEngineIndex() throws Exception {
        Directory dir = newFSDirectory(createTempDir());
        String segmentName = "_dummy";
        String indexNmsLibPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                String.format("%s_%s%s", segmentName, KNNEngine.NMSLIB.getName(),
                        KNNEngine.NMSLIB.getExtension())).toString();
        String indexFaissPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                String.format("%s_%s%s", segmentName, KNNEngine.FAISS.getName(),
                        KNNEngine.FAISS.getExtension())).toString();
        {
            String[] algoParams = {};
            AccessController.doPrivileged(
                    (PrivilegedAction<Void>) () -> {
                        KNNNmsLibIndex.save(docs, vectors, indexNmsLibPath, algoParams, SpaceType.L2.getValue(),
                                METHOD_HNSW);
                        return null;
                    }
            );
        }
        {
            AccessController.doPrivileged(
                    (PrivilegedAction<Void>) () -> {
                        KNNFaissIndex.save(docs, vectors, indexFaissPath, Collections.emptyMap(),
                                SpaceType.L2.getValue(), "Flat",
                                MIN_TRAINING_DATASET_SIZE_LIMIT + 1, MIN_MINIMUM_DATAPOINTS + 1);
                        return null;
                    }
            );
        }
        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy_NMSLIB" + KNNEngine.NMSLIB.getExtension()));
        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy_FAISS" + KNNEngine.FAISS.getExtension()));

        float[] queryVector = {1.0f, 1.0f, 1.0f, 1.0f};
        String[] algoQueryParams = {"efSearch=20"};

        {
            final KNNNmsLibIndex knnNmsLibIndex = KNNNmsLibIndex.load(indexNmsLibPath, algoQueryParams, SpaceType.L2);
            final KNNQueryResult[] results = knnNmsLibIndex.query(queryVector, 30);
            Map<Integer, Float> scores = Arrays.stream(results).collect(
                    Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
            assertEquals(results.length, 3);
            assertEquals(14.0, scores.get(0), 0.001);
            assertEquals(126.0, scores.get(1), 0.001);
            assertEquals(366.0, scores.get(2), 0.001);
        }
        {
            final KNNFaissIndex knnFaissLibIndex = KNNFaissIndex.load(indexFaissPath, SpaceType.L2);
            final KNNQueryResult[] results = knnFaissLibIndex.query(queryVector, 30);
            Map<Integer, Float> scores = Arrays.stream(results).collect(
                    Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
            logger.info(scores);
            assertEquals(results.length, 3);
            assertEquals(14.0, scores.get(0), 0.001);
            assertEquals(126.0, scores.get(1), 0.001);
            assertEquals(366.0, scores.get(2), 0.001);
        }
        dir.close();
    }

    public void testQueryMixedEngineWithWrongEngine() throws Exception {
        int[] docs = {0, 1, 2};

        float[][] vectors = {
                {5.0f, 6.0f, 7.0f, 8.0f},
                {1.0f, 2.0f, 3.0f, 4.0f},
                {9.0f, 10.0f, 11.0f, 12.0f}
        };

        Directory dir = newFSDirectory(createTempDir());
        String segmentName = "_dummy";
        String indexNmsLibPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                String.format("%s_%s%s", segmentName, KNNEngine.NMSLIB.getName(),
                        KNNEngine.NMSLIB.getExtension())).toString();
        String indexFaissPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                String.format("%s_%s%s", segmentName, KNNEngine.FAISS.getName(),
                        KNNEngine.FAISS.getExtension())).toString();
        {
            String[] algoParams = {};
            KNNNmsLibIndex.save(docs, vectors, indexNmsLibPath, algoParams, SpaceType.L2.getValue(), METHOD_HNSW);
        }
        {
            KNNFaissIndex.save(docs, vectors, indexFaissPath, Collections.emptyMap(),
                    SpaceType.L2.getValue(), "Flat",
                    MIN_TRAINING_DATASET_SIZE_LIMIT + 1, MIN_MINIMUM_DATAPOINTS + 1);
        }
        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy_NMSLIB" + KNNEngine.NMSLIB.getExtension()));
        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy_FAISS" + KNNEngine.FAISS.getExtension()));

        float[] queryVector = {1.0f, 1.0f, 1.0f, 1.0f};
        String[] algoQueryParams = {"efSearch=20"};

        {
            //Load Index With Wrong Engine, need throw Exception
            expectThrows(Exception.class, () -> KNNFaissIndex.load(indexNmsLibPath, SpaceType.L2));
            final KNNNmsLibIndex knnNmsLibIndex = KNNNmsLibIndex.load(indexNmsLibPath, algoQueryParams, SpaceType.L2);
            final KNNQueryResult[] results = knnNmsLibIndex.query(queryVector, 30);
            Map<Integer, Float> scores = Arrays.stream(results).collect(
                    Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
            logger.info(scores);
            assertEquals(results.length, 3);
            assertEquals(126.0, scores.get(0), 0.001);
            assertEquals(14.0, scores.get(1), 0.001);
            assertEquals(366.0, scores.get(2), 0.001);
        }
        {
            //Load Index With Wrong Engine, need throw Exception
            expectThrows(Exception.class, () -> KNNNmsLibIndex.load(indexFaissPath, algoQueryParams, SpaceType.L2));
            final KNNFaissIndex knnFaissLibIndex = KNNFaissIndex.load(indexFaissPath, SpaceType.L2);
            final KNNQueryResult[] results = knnFaissLibIndex.query(queryVector, 30);
            Map<Integer, Float> scores = Arrays.stream(results).collect(
                    Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
            logger.info(scores);
            assertEquals(results.length, 3);
            assertEquals(126.0, scores.get(0), 0.001);
            assertEquals(14.0, scores.get(1), 0.001);
            assertEquals(366.0, scores.get(2), 0.001);
        }
        dir.close();
    }

    public void testQueryHnswIndexWithValidAlgoParams() throws Exception {
        int[] docs = {0, 1, 2};

        float[][] vectors = {
                {5.0f, 6.0f, 7.0f, 8.0f},
                {1.0f, 2.0f, 3.0f, 4.0f},
                {9.0f, 10.0f, 11.0f, 12.0f}
        };

        Directory dir = newFSDirectory(createTempDir());
        String indexPath = getIndexPath(dir);

        /**
         * Passing valid algo params should not fail the graph construction.
         */
        String[] algoIndexParams = {"M=32","efConstruction=200"};
        AccessController.doPrivileged(
                (PrivilegedAction<Void>) () -> {
                    KNNNmsLibIndex.save(docs, vectors, indexPath, algoIndexParams, SpaceType.L2.getValue(),
                            METHOD_HNSW);
                    return null;
                }
        );


        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy1.hnsw"));

        float[] queryVector = {1.0f, 1.0f, 1.0f, 1.0f};
        String[] algoQueryParams = {"efSearch=200"};

        final KNNIndex index = KNNNmsLibIndex.load(indexPath, algoQueryParams, SpaceType.L2);
        final KNNQueryResult[] results = index.query(queryVector, 30);

        Map<Integer, Float> scores = Arrays.stream(results).collect(
                Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
        logger.info(scores);

        assertEquals(results.length, 3);
        /*
         * scores are evaluated using Euclidean distance. Distance of the documents with
         * respect to query vector are as follows
         * doc0 = 126, doc1 = 14,  doc2 = 366
         * Nearest neighbor is doc1 then doc0 then doc2
         */
        assertEquals(126.0, scores.get(0), 0.001);
        assertEquals(14.0, scores.get(1), 0.001);
        assertEquals(366.0, scores.get(2), 0.001);
        dir.close();
    }

    public void testAddAndQueryHnswIndexInnerProd() throws Exception {
        int[] docs = {0, 1, 2};

        float[][] vectors = {
                {1.0f, -1.0f},
                {-1.0f, 1.0f},
                {0.0f, 0.0f}
        };

        Directory dir = newFSDirectory(createTempDir());
        String segmentName = "_dummy1";
        String indexPath = Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                String.format("%s.hnsw", segmentName)).toString();

        String[] algoParams = {};
        AccessController.doPrivileged(
                (PrivilegedAction<Void>) () -> {
                    KNNNmsLibIndex.save(docs, vectors, indexPath, algoParams, SpaceType.INNER_PRODUCT.getValue(),
                            METHOD_HNSW);
                    return null;
                }
        );

        assertTrue(Arrays.asList(dir.listAll()).contains("_dummy1.hnsw"));

        float[] queryVector = {2.0f, -2.0f};
        String[] algoQueryParams = {"efSearch=20"};

        final KNNIndex knnIndex = KNNNmsLibIndex.load(indexPath, algoQueryParams, SpaceType.INNER_PRODUCT);
        final KNNQueryResult[] results = knnIndex.query(queryVector, 30);

        Map<Integer, Float> scores = Arrays.stream(results).collect(
                Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
        logger.info(scores);

        assertEquals(results.length, 3);
        /*
         * scores are evaluated using negative dot product similarity. Distance of the documents with
         * respect to query vector are as follows
         * doc0 = -4.0, doc1 = 4.0,  doc2 = 0.0
         * Nearest neighbor is doc1 then doc0 then doc2
         */
        assertEquals(scores.get(0), -4.0, 1e-4);
        assertEquals(scores.get(1), 4.0, 1e-4);
        assertEquals(scores.get(2), 0.0, 1e-4);
        dir.close();
    }

    private String getIndexPath(Directory dir ) {
        String segmentName = "_dummy1";
        return Paths.get(((FSDirectory) (FilterDirectory.unwrap(dir))).getDirectory().toString(),
                String.format("%s.hnsw", segmentName)).toString();
    }
}
