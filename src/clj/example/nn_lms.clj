(ns example.nn-lms
  (:require [neuroph-clj.core :as network])
  (:import [org.neuroph.core Layer NeuralNetwork]
           [org.neuroph.core.data DataSet DataSetRow]
           [org.neuroph.nnet.comp.neuron BiasNeuron InputNeuron]
           [org.neuroph.nnet.learning BackPropagation LMS]
           [org.neuroph.util ConnectionFactory LayerFactory NeuralNetworkFactory NeuralNetworkType NeuronProperties TransferFunctionType]))

(defn create-network
  "构建神经网络"
  [input-neurons-count]
    (let [[input-layer output-layer :as layers] (network/create-layers [2 1] [{:neuronType InputNeuron} {:transferFunction TransferFunctionType/STEP}])
          nn (network/create-neural-network network/PERCEPTRON-TYPE layers)]
      nn
      ))

(defn test-neural-network
  [neural-net data-set]
  (doseq [test-set-row (.getRows data-set)]
    (let [input (.getInput test-set-row)
          neural-net (doto neural-net (.setInput input)
                           (.calculate))
          output (.getOutput neural-net)]
        (print "Input:" (vec input))
        (println "Output:" (vec output)))))

(defn run
  "运行神经网络"
  []
  (let [data-set-rows (network/create-data-set-rows ["0 0" "0 1" "1 0" "1 1"] ["0" "0" "0" "1"])
        training-set (network/create-data-set 2 1 data-set-rows)
        my-perceptron (create-network 2)]
    (do
      (println "training neural network ......")
      (network/learn my-perceptron (LMS.) training-set)
      (println "testing training neural network")
      (test-neural-network my-perceptron training-set))))
