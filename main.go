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

	toolParams := toToolParams(toolDefs)
	tools := make([]anthropic.ToolUnionParam, len(toolParams))
	for i, toolParam := range toolParams {
		tools[i] = anthropic.ToolUnionParam{OfTool: &toolParam}
	}

	agent := Agent{client: client, getInput: getInput, tools: tools}
	agent.Run()

}

type Agent struct {
	client   anthropic.Client
	getInput func() (string, bool)
	tools    []anthropic.ToolUnionParam
}

func (a *Agent) Run() {
	fmt.Println("Get started with a request.")

	var messages []anthropic.MessageParam

	for {
		input, ok := a.getInput()
		if ok {
			fmt.Println("Got input:", input)
			messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(input)))

			for {
				message := a.runInference(messages)

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

				messages = append(messages, message.ToParam())
				toolResults := []anthropic.ContentBlockParamUnion{}

				for _, block := range message.Content {
					switch variant := block.AsAny().(type) {
					case anthropic.ToolUseBlock:
						print("[user (" + block.Name + ")]: ")
						toolDef, ok := toolDefs[block.Name]
						if !ok {
							panic(fmt.Sprintf("Tool is not in toolDefs map: %s", block.Name))
						}

						response, err := toolDef.Fn(json.RawMessage([]byte(variant.JSON.Input.Raw())))
						if err != nil {
							panic("Tool use error.")
						}

						b, err := json.Marshal(response)
						if err != nil {
							panic(err)
						}

						println(string(b))

						toolResults = append(toolResults, anthropic.NewToolResultBlock(block.ID, string(b), false))
					}

				}
				if len(toolResults) == 0 {
					break
				}
				messages = append(messages, anthropic.NewUserMessage(toolResults...))
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
		Tools:     a.tools,
	})
	if err != nil {
		panic(err)
	}
	return message
}

type ToolDef struct {
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Fn          func(input json.RawMessage) (any, error)
}

var getCoordinatesTool = ToolDef{
	Description: "Accepts a place as an address, then returns the latitude and longitude coordinates.",
	InputSchema: anthropic.ToolInputSchemaParam{
		Properties: map[string]interface{}{
			"location": map[string]interface{}{
				"type":        "string",
				"description": "The name of the place or address to get coordinates for.",
			},
		},
		Required: []string{"location"},
	},

	Fn: func(raw json.RawMessage) (any, error) {
		var input struct {
			Location string `json:"location"`
		}

		err := json.Unmarshal(raw, &input)
		if err != nil {
			panic(err)
		}

		return map[string]int{"lat": 1, "long": 2}, nil
	},
}

var toolDefs = map[string]ToolDef{
	"get_coordinates": getCoordinatesTool,
}

func toToolParams(toolDefs map[string]ToolDef) []anthropic.ToolParam {
	var tools []anthropic.ToolParam
	for toolName, toolDef := range toolDefs {
		tools = append(tools, anthropic.ToolParam{
			Name:        toolName,
			Description: anthropic.String(toolDef.Description),
			InputSchema: toolDef.InputSchema,
		})
	}
	return tools
}
