import React, { useRef, useState, useEffect } from "react";
import { getOrCreateUserId } from "../userId";
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
  const [userId, setUserId] = useState()

  useEffect(() => {
    const userId = getOrCreateUserId();
    console.log("Visitor ID:", userId);
    setUserId(userId)
  }, [])

  const handlePredict = async () => {
    setShape("...");
    const image_data = await canvasRef.current.exportImage("png");

    const body = {
      image_b64: image_data.split(",")[1],
      user_id: userId,
    };

    // Make POST request to send data
    axios
        .post("http://127.0.0.1:8080/predict", body)
        .then((response) => {
            console.log(response.data.label);
            setShape(response.data.label)
            setConfidence(response.data.confidence)
        })
        .catch((error) => {
            console.error('Error:', error.message);
            try {
              // Attempt to parse the error message if it's a stringified JSON
              const errorDetails = JSON.parse(error.message);
              console.log('422 Error Details:', errorDetails);
              // You can then access specific properties of the errorDetails object
              // For example: errorDetails.message, errorDetails.errors, etc.
            } catch (parseError) {
              console.log('Could not parse error message as JSON:', parseError);
            }
        });
  };

  const handleTrain = async (category) => {
    const image_data = await canvasRef.current.exportImage("png");

    const body = {
      image_b64: image_data.split(",")[1],
      category: category,
      user_id: userId,
    };

    // Make POST request to send data
    axios
        .post("http://127.0.0.1:8080/train", body)
        .then((response) => {
            console.log(response.data);
            // setShape(response.data.label)
            handleClear();
        })
        .catch((error) => {
            console.error('Error:', error.message);
            try {
              // Attempt to parse the error message if it's a stringified JSON
              const errorDetails = JSON.parse(error.message);
              console.log('422 Error Details:', errorDetails);
              // You can then access specific properties of the errorDetails object
              // For example: errorDetails.message, errorDetails.errors, etc.
            } catch (parseError) {
              console.log('Could not parse error message as JSON:', parseError);
            }
        });
  };

  const handleSave = async () => {
    const image_data = await canvasRef.current.exportImage("png");

    const body = {
      image_b64: image_data.split(",")[1],
    };

    // Make POST request to send data
    axios
        .post("http://127.0.0.1:8080/save", body)
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
