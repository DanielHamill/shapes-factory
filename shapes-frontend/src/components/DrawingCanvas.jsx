import React, { useRef, useState } from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";
import axios from 'axios';
import ProgressBar from "@ramonak/react-progress-bar";

const styles = {
  border: "0.0625rem solid #9c9c9c",
  borderRadius: "0.25rem",
};

export default function DrawingCanvas() {
  const canvasRef = useRef();
  const [shape, setShape] = useState("")
  const [confidence, setConfidence] = useState(0)

  const handlePredict = async () => {
    setShape("...");
    const image_data = await canvasRef.current.exportImage("png");

    const body = {
      image_b64: image_data.split(",")[1],
    };

    // Make POST request to send data
    axios
        .post("http://127.0.0.1:8000/predict", body)
        .then((response) => {
            console.log(response.data.label);
            setShape(response.data.label)
            setConfidence(response.data.confidence)
        })
        .catch((err) => {
            console.log("Error creating post");
        });
  };

  const handleTrain = async (category) => {
    const image_data = await canvasRef.current.exportImage("png");

    const body = {
      image_b64: image_data.split(",")[1],
      category: category,
    };

    // Make POST request to send data
    axios
        .post("http://127.0.0.1:8000/train", body)
        .then((response) => {
            console.log(response.data);
            // setShape(response.data.label)
            handleClear();
        })
        .catch((err) => {
            console.log("Error creating post");
        });
  };

  const handleSave = async () => {
    const image_data = await canvasRef.current.exportImage("png");

    const body = {
      image_b64: image_data.split(",")[1],
    };

    // Make POST request to send data
    axios
        .post("http://127.0.0.1:8000/save", body)
        .then((response) => {
            console.log(response.data.label);
            setShape(response.data.label);
            handleClear();
        })
        .catch((err) => {
            console.log("Error creating post");
        });
  };

  const handleClear = () => {
    canvasRef.current.clearCanvas();
  };

  return (
    <div>
      <ReactSketchCanvas
        ref={canvasRef}
        style={styles}
        width="400px"
        height="400px"
        strokeWidth={15}
        strokeColor="black"
      />
      <div>
        I'm not very confident in this prediction yet :(
      </div>
      <div>
        Confidence:
        <ProgressBar completed={Math.round(confidence)} maxCompleted={100} width="400px"/>
      </div>
      <div style={{ marginTop: "1rem" }}>
        <button onClick={handlePredict}>
          Predict
        </button>
        <button onClick={handleSave} style={{ marginLeft: "1rem" }}>
          Save
        </button>
        <button onClick={handleClear} style={{ marginLeft: "1rem" }}>
          Clear
        </button>
        <button onClick={() => handleTrain(0)} style={{ marginLeft: "1rem" }}>
          Train 0
        </button>
        <button onClick={() => handleTrain(1)} style={{ marginLeft: "1rem" }}>
          Train 1
        </button>
        <div>
          Prediction: {shape}
        </div>
      </div>
    </div>
  );
}
