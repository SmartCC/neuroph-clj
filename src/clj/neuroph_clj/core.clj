(ns neuroph-clj.core
  (:import [org.neuroph.core NeuralNetwork]
           [org.neuroph.core.data DataSet DataSetRow]
           [org.neuroph.core.transfer Linear]
           [org.neuroph.nnet.comp.neuron BiasNeuron]
           [org.neuroph.util ConnectionFactory LayerFactory NeuralNetworkFactory NeuralNetworkType NeuronProperties TransferFunctionType]
           [java.io File FileInputStream]))

;神经网络的类型
(def ADALINE-TYPE NeuralNetworkType/ADALINE)
(def BAM-TYPE NeuralNetworkType/BAM)
(def BOLTZMAN-TYPE NeuralNetworkType/BOLTZMAN)
(def COMPETITIVEE-TYPE NeuralNetworkType/COMPETITIVE)
(def COUNTERPROPAGATION-TYPE NeuralNetworkType/COUNTERPROPAGATION)
(def HOPFIELD-TYPE NeuralNetworkType/HOPFIELD)
(def INSTAR-TYPE NeuralNetworkType/INSTAR)
(def INSTAR-OUTSTAR-TYPE NeuralNetworkType/INSTAR_OUTSTAR)
(def KOHONEN-TYPE NeuralNetworkType/KOHONEN)
(def MAXNET-TYPE NeuralNetworkType/MAXNET)
(def MULTI-LAYER-PERCEPTRON-TYPE NeuralNetworkType/MULTI_LAYER_PERCEPTRON)
(def NEURO-FUZZY-REASONER-TYPE NeuralNetworkType/NEURO_FUZZY_REASONER)
(def OUTSTAR-TYPE NeuralNetworkType/OUTSTAR)
(def PCA-NETWORK-TYPE NeuralNetworkType/PCA_NETWORK)
(def PERCEPTRON-TYPE NeuralNetworkType/PERCEPTRON)
(def RBF-NETWORK-TYPE NeuralNetworkType/RBF_NETWORK)
(def RECOMMENDER-TYPE NeuralNetworkType/RECOMMENDER)
(def SUPERVISED-HEBBIAN-NET-TYPE NeuralNetworkType/SUPERVISED_HEBBIAN_NET)
(def UNSUPERVISED-HEBBIAN-NET-TYPE NeuralNetworkType/UNSUPERVISED_HEBBIAN_NET)


;转化函数的类型
(def GAUSSIAN-TYPE TransferFunctionType/LINEAR)
(def LINEAR-TYPE TransferFunctionType/LINEAR)
(def LOG-TYPE TransferFunctionType/LOG)
(def RAMP-TYPE TransferFunctionType/RAMP)
(def SGN-TYPE TransferFunctionType/SGN)
(def SIGMOID-TYPE TransferFunctionType/SIGMOID)
(def SIN-TYPE TransferFunctionType/SIN)
(def STEP-TYPE TransferFunctionType/STEP)
(def TANH-TYPE TransferFunctionType/TANH)
(def TRAPEZOID-TYPE TransferFunctionType/TRAPEZOID)

(defn create-data-set-rows
  "创建DataSetRow，添加的类型为string类型或者[D类型，如果时string 类型以空格分割，如 \"2 3\""
  [input-seq output-seq]
  {:pre [(or (and (every? string? input-seq)
                  (every? string? output-seq))
             (and (every? #(= (Class/forName "[D") (class %)) input-seq)
                  (every? #(= (Class/forName "[D") (class %)) output-seq)))]}
   (map #(DataSetRow. % %2) input-seq output-seq))

(defn add-rows-2-data-set
  "添加行元素进DataSet"
  ([*data-set* data-set-rows]
   (doseq [row data-set-rows]
     (.addRow *data-set* row)))
  ([*data-set* input-seq output-seq]
   {:pre [(and (every? #(= (Class/forName "[D") (class %)) input-seq)
               (every? #(= (Class/forName "[D") (class %)) output-seq))]}
   (doseq [in input-seq o output-seq]
     (.addRow *data-set* in o))))

(defn create-data-set
  "创建一个data-set并把rows数据填充到里边"
  ([input-num output-num data-set-rows]
   (doto (DataSet. input-num output-num)
     (add-rows-2-data-set data-set-rows)))
  ([input-num input-seq output-num output-seq]
   (doto (DataSet. input-num output-num)
     (add-rows-2-data-set (create-data-set-rows input-seq output-seq)))))

 
(defn create-neuron-properties
  "创建neuron properties"
  [properties]
  {:pre [(map? properties)]}
  (doto (NeuronProperties.)
    (#(doseq [property properties] (.setProperty % (name (key property)) (val property))))))

(defn create-layer
  "创建layer"
  [input-num properties]
  (LayerFactory/createLayer input-num (create-neuron-properties properties)))

(defn add-bias-neuron-2-layer
  "添加偏置"
  [*layer*]
  (.addNeuron *layer* (BiasNeuron.))
  *layer*)

(defn create-layers
  "创建layers"
  [[input-layer-num output-layer-num & middle-layers-num] [input-neuron-properties output-neuron-properties middle-neuron-properties]]
  (let [input-layer (create-layer input-layer-num input-neuron-properties)
        output-layer (create-layer output-layer-num output-neuron-properties)
        middle-layers (map create-layer middle-layers-num (repeat middle-neuron-properties))
        ]
  (concat [(add-bias-neuron-2-layer input-layer)]
          (map add-bias-neuron-2-layer middle-layers)
          [output-layer])))  

(defn create-layers-connect
  "将layer加入神经网并连接各layer"
  [*nn* *layers*]
  (doseq [layer *layers*]
    (.addLayer *nn* layer))
  (reduce #(do (ConnectionFactory/fullConnect % %2) (identity %2)) *layers*))

(defn create-neural-network
  [nn-type *layers*]
  (doto (NeuralNetwork.)
    (.setNetworkType nn-type)
    (create-layers-connect *layers*)))

(defn learn
  [*nn* learning-rule traing-set]
  (doto *nn*
    (.setLearningRule learning-rule)
    (NeuralNetworkFactory/setDefaultIO)
    (.learn traing-set)))

(defn run-nn
  [*nn* input]
  {:pre [(= (Class/forName "[D") (class input))]}
  (.getOutput (doto *nn* (.setInput input) (.calculate))))
