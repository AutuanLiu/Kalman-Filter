workflow "New workflow" {
  on = "push"
  resolves = ["Create an issue"]
}

action "GitHub Action for Azure" {
  uses = "Azure/github-actions/cli@cb630b3bef716c326bbcc3a1e47623254ae82dd3"
}

action "Create an issue" {
  uses = "JasonEtco/create-an-issue@d10d7bc2a567fa4288ead6b91f307aa4b44fb9f7"
  needs = ["GitHub Action for Azure"]
}
