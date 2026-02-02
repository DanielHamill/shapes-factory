import { ReactSketchCanvas } from "react-sketch-canvas";
import { Card, Flex, Heading } from "@radix-ui/themes";

const styles = {
  border: "0.0625rem solid #9c9c9c",
  borderRadius: "0px",
};

export default function DrawingCanvas({ canvasRef }) {
  return (
    <Card Shadow="6">
        <Flex direction="column" gap="4" padding="4" align="center">
            <Heading as="h1">Drawing Canvas</Heading>
            <ReactSketchCanvas
                ref={canvasRef}
                style={styles}
                width="400px"
                height="400px"
                strokeWidth={15}
                strokeColor="black"
            />
        </Flex>
      
    </Card>
  );
}
