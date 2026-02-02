import CanvasPanel from './components/CanvasPanel.jsx';
import InfoPanel from './components/InfoPanel.jsx';
import InputsPanel from './components/InputsPanel.jsx';
import { Flex } from '@radix-ui/themes';
import "./App.css"

function App() {
  return (
    <Flex gap="4" m="auto" justify="center" mt="4" wrap="wrap">
      <CanvasPanel />
      <Flex direction="column" gap="4">
        <InfoPanel />
        <InputsPanel />
      </Flex>
    </Flex>
    
  );
}

export default App;
