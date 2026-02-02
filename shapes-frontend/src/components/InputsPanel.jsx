import { Card, Grid, Button, Text, Flex } from "@radix-ui/themes";

export default function InputsPanel({ prediction }) {
    const buttonVariant = "soft"
    return (
        <Card>
            <Grid columns="3" rows="2" gap="2" padding="4">
                <Button variant={buttonVariant}>Predict</Button>
                <Button variant={buttonVariant}>Save</Button>
                <Button variant={buttonVariant}>Clear</Button>
                <Button variant={buttonVariant}>Train 0</Button>
                <Button variant={buttonVariant}>Train 1</Button>
                <Button variant={buttonVariant}>Train 2</Button>
            </Grid>
            <Flex align="center" justify="center" padding="4" mt="4">
                <Text size="3">Prediction: {prediction}</Text>
            </Flex>
        </Card>
    )
}