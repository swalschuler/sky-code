package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()

	scanner := bufio.NewScanner(os.Stdin)
	getInput := func() (string, bool) {
		if scanner.Scan() {
			return scanner.Text(), true
		}
		return "", false
	}

	agent := Agent{client: client, getInput: getInput}
	agent.Run()

}

type Agent struct {
	client   anthropic.Client
	getInput func() (string, bool)
}

func (a *Agent) Run() {
	fmt.Println("Get started with a request.")

	var messages []anthropic.MessageParam

	for {
		input, ok := a.getInput()
		if ok {
			fmt.Println("Got input:", input)
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(input)))
			message := a.runInference(messages)
			messages = append(messages, message.ToParam())

			print("[assistant]: ")
			for _, block := range message.Content {
				switch block := block.AsAny().(type) {
				case anthropic.TextBlock:
					println(block.Text)
					println()
				case anthropic.ToolUseBlock:
					inputJSON, _ := json.Marshal(block.Input)
					println(block.Name + ": " + string(inputJSON))
					println()
				}
			}

		} else {
			fmt.Println("End of input.")
		}

	}
}

func (a *Agent) runInference(messages []anthropic.MessageParam) *anthropic.Message {
	message, err := a.client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeOpus4_6,
		Messages:  messages,
		MaxTokens: 1024,
	})
	if err != nil {
		panic(err)
	}
	return message
}
