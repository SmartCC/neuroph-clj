(ns example.pos-or-neg
  (:import [java.io File FileInputStream]
           [java.util Random]
           [org.neuroph.core NeuralNetwork]
           [org.neuroph.core.data DataSet]
           [org.neuroph.core.transfer Linear Step]
           [org.neuroph.nnet.comp.neuron BiasNeuron InputNeuron]
           [org.neuroph.nnet.learning BackPropagation MomentumBackpropagation]
           [org.neuroph.util ConnectionFactory LayerFactory NeuralNetworkFactory NeuralNetworkType NeuronProperties TransferFunctionType]
           [org.neuroph.util.random NguyenWidrowRandomizer])
  (:use [neuroph-clj.core]))

(def zero (double-array [0 0 0 0])) ;零
(def pe (double-array [0 0 0 1])) ;正偶数
(def ne (double-array [0 0 1 0])) ;负偶数
(def po (double-array [0 1 0 0])) ;正奇数
(def no (double-array [1 0 0 0])) ;负奇数

(defn correct-classify
  "检验数字的正负和奇偶"
  [i]
  (if (zero? i) zero
      (if (> i 0)
        (if (zero? (mod i 2)) pe po)
        (if (zero? (mod i 2)) ne no))))

(defn int-2-double-array
  "将整数转化为二进制double数组"
  [i]
  (loop [j 0 res []]
    (if (>= j 32)
      (double-array res)
      (recur (inc j) (conj res (double (bit-and (bit-shift-right i j) 1)))))))

(defn add-layer-create-connect
  [network prev-layer layer]
  (do
    (.addLayer network layer)
    (ConnectionFactory/fullConnect prev-layer layer)))

(defn create-middle-layer
  [network prev-layer neurons-num neuron-properties]
  (loop [neurons-num neurons-num prev-layer prev-layer]
      (if (empty? neurons-num)
        prev-layer
        (let [layer-tmp (doto (LayerFactory/createLayer (first neurons-num) neuron-properties)
                          (.addNeuron (BiasNeuron.)))]
          (add-layer-create-connect network prev-layer layer-tmp)
          (recur (next neurons-num) layer-tmp)))))

(defn create-random-int-seq
  "创建随机整数序列，用于训练和测试神经网络"
  [int-num]
  (let [r (Random.)]
    (repeatedly int-num #(.nextInt r))))

(defn test-neural-network
  "测试神经网络模型"
  [neural-network]
  (let [test-num 2000
        test-ints (create-random-int-seq test-num)]
    (->> (map #(= (vec (correct-classify %))
                  (do (.setInput neural-network (int-2-double-array %))
                      (.calculate neural-network)
                      (vec (.getOutput neural-network))))
              test-ints)
         (filter identity)
         count
         (#(* 100 (/ % test-num)))
         double)))

(defn run-model
  "训练模型"
  []
  (let [train-num 2000
        train-set (DataSet. 32 4)
        rand-ints (create-random-int-seq train-num)
        *layers* (create-layers [32 4 4] [{:neuronType InputNeuron :transferFunction Linear} {:transferFunction Step} {:transferFunction TransferFunctionType/SIGMOID}])
        nn (create-neural-network  NeuralNetworkType/MULTI_LAYER_PERCEPTRON *layers*)
        ]
    (do
      (doseq [i rand-ints]
        (.addRow train-set (int-2-double-array i) (correct-classify i)))
      (learn nn (MomentumBackpropagation.) train-set)
      (.save nn "posOrNeg1.nnet"))))

(defn run-test
  "测试模型
  userage: run [model-file-path]"
  [model-file-path]
  (test-neural-network (NeuralNetwork/load (FileInputStream. (File. model-file-path)))))
