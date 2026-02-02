import { Card, Flex, Text } from "@radix-ui/themes";
import ProgressBar from "@ramonak/react-progress-bar";
import React from 'react';


export default function InfoPanel({confidence}) {
    const [status, setStatus] = React.useState("I'm not very confident in this prediction yet :(");

    return (
        <Card>
            <Flex direction="column" minWidth="200px" gap="4" padding="4" align="center">
                <Text size="3" wrap="wrap">{status}</Text>
                <ProgressBar 
                    completed={Math.round(confidence)} 
                    maxCompleted={100} 
                    width="400px"
                    bgColor="var(--accent-9)"
                    baseBgColor="var(--accent-3)"
                />
            </Flex>
        </Card>
    )
}