import CanvasPanel from './components/CanvasPanel.jsx';
import InfoPanel from './components/InfoPanel.jsx';
import InputsPanel from './components/InputsPanel.jsx';
import { Flex } from '@radix-ui/themes';
import { useRef, useState, useEffect } from "react";
import { getOrCreateUserId } from "./userId";
import "./App.css"

function App() {
  const canvasRef = useRef();
  const [shape, setShape] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [userId, setUserId] = useState();

  useEffect(() => {
    const userId = getOrCreateUserId();
    console.log("Visitor ID:", userId);
    setUserId(userId);
  }, []);

  return (
    <Flex gap="4" m="auto" justify="center" mt="4" wrap="wrap">
      <CanvasPanel canvasRef={canvasRef} />
      <Flex direction="column" gap="4">
        <InfoPanel confidence={confidence} />
        <InputsPanel 
          prediction={shape}
          canvasRef={canvasRef}
          userId={userId}
          setShape={setShape}
          setConfidence={setConfidence}
        />
      </Flex>
    </Flex>
    
  );
}

export default App;
