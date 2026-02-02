import { Card, Grid, Button, Text, Flex } from "@radix-ui/themes";
import axios from 'axios';

export default function InputsPanel({ prediction, canvasRef, userId, setShape, setConfidence }) {
    const buttonVariant = "soft"
    
    const handlePredict = async () => {
        setShape("...");
        const image_data = await canvasRef.current.exportImage("png");

        const body = {
            image_b64: image_data.split(",")[1],
            user_id: userId,
        };

        // Make POST request to send data
        axios
            .post(`${process.env.REACT_APP_BACKEND_URL}/predict`, body)
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
            .post(`${process.env.REACT_APP_BACKEND_URL}/train`, body)
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
            .post(`${process.env.REACT_APP_BACKEND_URL}/save`, body)
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
        <Card>
            <Grid columns="3" rows="2" gap="2" padding="4">
                <Button variant={buttonVariant} onClick={handlePredict}>Predict</Button>
                <Button variant={buttonVariant} onClick={handleSave}>Save</Button>
                <Button variant={buttonVariant} onClick={handleClear}>Clear</Button>
                <Button variant={buttonVariant} onClick={() => handleTrain(0)}>Train 0</Button>
                <Button variant={buttonVariant} onClick={() => handleTrain(1)}>Train 1</Button>
                <Button variant={buttonVariant} onClick={() => handleTrain(2)}>Train 2</Button>
            </Grid>
            <Flex align="center" justify="center" padding="4" mt="4">
                <Text size="3">Prediction: {prediction}</Text>
            </Flex>
        </Card>
    )
}