package tex

import "testing"

func TestValidateGood(t *testing.T) {
	r := Validate(`\frac{a}{b} + \sqrt{x^2+y^2}`)
	if !r.Valid {
		t.Fatalf("expected valid: %+v", r.Checks)
	}
}
func TestValidateBad(t *testing.T) {
	r := Validate(`\frac{a}{b`)
	if r.Valid {
		t.Fatal("expected invalid (unbalanced brace)")
	}
}
func TestRepair(t *testing.T) {
	r := Repair(`\frac{a}{b`)
	if !r.Changed {
		t.Fatal("expected changed")
	}
	if !r.Valid {
		t.Fatalf("repair should produce valid latex: %+v", r)
	}
	if r.Latex != `\frac{a}{b}` {
		t.Fatalf("unexpected repair: %q", r.Latex)
	}
}

func TestRepairExtraBraces(t *testing.T) {
	cases := map[string]string{
		`\frac{a}{b}}`:   `\frac{a}{b}`,
		`}x = 1`:         `x = 1`,
		`}x = 1}`:        `x = 1`,
		`x = 1}}`:        `x = 1`,
		`\frac{a}{b}} x`: `\frac{a}{b} x`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
		if !r.Valid {
			t.Fatalf("repair(%q) should be valid: %+v", in, r)
		}
	}
	// All-brace input becomes empty → correctly invalid.
	r := Repair(`}}}`)
	if r.Latex != "" {
		t.Fatalf("repair(`}}}`) = %q, want empty", r.Latex)
	}
	if r.Valid {
		t.Fatal("empty result should be invalid")
	}
}
func TestValidateProse(t *testing.T) {
	r := Validate("hello this is some prose words")
	for _, c := range r.Checks {
		if c.Type == "content" && c.OK {
			t.Fatalf("prose should fail content check")
		}
	}
}
func TestValidateTextContent(t *testing.T) {
	cases := []string{
		`\text{已知 $x^2$}`,
		`\text{当 a \geq b 时}`,
		`\text{面积 2x+3=7}`,
	}
	for _, in := range cases {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "text" && c.OK {
				t.Fatalf("text check should fail on latex inside \\text{}: %q (%s)", in, c.Detail)
			}
		}
	}
	good := []string{`\text{已知 x 的平方}`, `\text{图 2}`, `\text{hello world}`, `\text{第一问-第二问}`, `\text{面积约 10 平方}`}
	for _, in := range good {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "text" && !c.OK {
				t.Fatalf("text check should pass on plain text: %q (%s)", in, c.Detail)
			}
		}
	}
}
func TestRepairTextContent(t *testing.T) {
	cases := map[string]string{
		`\text{已知 2x+3=7}`:      `\text{已知 }$2x+3=7$`,
		`\text{已知 $x^2$}`:       `\text{已知 }$x^2$`,
		`\text{当 $a \geq b$ 时}`: `\text{当 }$a \geq b$\text{ 时}`,
		`\text{面积 $S$ 是 5}`:     `\text{面积 }$S$\text{ 是 5}`,
		`\text{$x^2$}`:          `$x^2$`,
		`\text{已知 x 的平方 图 2}`:   `\text{已知 x 的平方 图 2}`,
		`\text{第一问-第二问}`:        `\text{第一问-第二问}`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairBalancedEnv(t *testing.T) {
	cases := map[string]string{
		`\begin{aligned} a+b \end{aligned}`:                  `\begin{aligned} a+b \end{aligned}`,
		`\begin{aligned} x &= 1 \\ y &= 2 \end{aligned}`:     `\begin{aligned} x &= 1 \\ y &= 2 \end{aligned}`,
		`\begin{aligned} a+b`:                                `\begin{aligned} a+b\end{aligned}`,
		`\begin{cases} x \\ y \end{cases} \begin{aligned} z`: `\begin{cases} x \\ y \end{cases} \begin{aligned} z\end{aligned}`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
		if !r.Valid {
			t.Fatalf("repair(%q) should be valid: %q (log=%v)", in, r.Latex, r.Log)
		}
	}
}

func TestRepairDollarSpans(t *testing.T) {
	cases := map[string]string{
		`\text{当 $x>0`:       `\text{当 }$x>0$`,
		`\text{当 $x>0 时}`:    `\text{当 }$x>0 时$`,
		`\text{a $x$ b}`:     `\text{a }$x$\text{ b}`,
		`\text{面积 $S$ 是 5}`:  `\text{面积 }$S$\text{ 是 5}`,
		`\text{当 $$x+1$$ 时}`: `\text{当 }$$x+1$$\text{ 时}`,
		`\text{当 $$x+1`:      `\text{当 }$$x+1$$`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairProseWords(t *testing.T) {
	cases := map[string]string{
		`\text{note 3x+1}`:    `\text{note }$3x+1$`,
		`\text{where x > 0}`:  `\text{where }$x > 0$`,
		`\text{Given x > 0}`:  `\text{Given }$x > 0$`,
		`\text{已知 AB = CD}`:   `\text{已知 }$AB = CD$`,
		`\text{已知 tan A = 3}`: `\text{已知 }$tan A = 3$`,
		`\text{a \geq b}`:     `$a \geq b$`,
		`\text{第一问-第二问}`:      `\text{第一问-第二问}`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestValidateContentConsistency(t *testing.T) {
	prose := []string{
		`hello this is some prose words`,
		`\text{hello this is some prose words}`,
	}
	for _, in := range prose {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "content" && c.OK {
				t.Fatalf("content check should fail on prose: %q (%s)", in, c.Detail)
			}
		}
	}
	math := []string{
		`\frac{a}{b} + \text{note}`,
		`\text{已知 }$x>0$\text{ 时}`,
		`已知 $x + y = 5$`,
		`\text{图 2}`,
	}
	for _, in := range math {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "content" && !c.OK {
				t.Fatalf("content check should pass on formula: %q (%s)", in, c.Detail)
			}
		}
	}
}

func TestValidateDollarBalance(t *testing.T) {
	good := []string{
		`x = $y$ + 1`,
		`$$x = y$$`,
		`$x$ $y$`,
		`$$A$$ $B$`,
		`price \$5`,
		`x = \$5 + \$6`,
		`x = 1`,
		`x = 1 \\`,
	}
	for _, in := range good {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "dollar" && !c.OK {
				t.Fatalf("dollar check should pass: %q (%s)", in, c.Detail)
			}
		}
	}
	bad := []string{
		`x = $y+1`,
		`price $5`,
		`x $ y`,
		`$`,
	}
	for _, in := range bad {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "dollar" && c.OK {
				t.Fatalf("dollar check should fail on unclosed $: %q", in)
			}
		}
	}
}

func TestValidateDanglingOperator(t *testing.T) {
	bad := []string{`x^2+`, `x^`, `x_`, `x > `}
	for _, in := range bad {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "structure" && c.OK {
				t.Fatalf("structure check should fail on dangling operator: %q", in)
			}
		}
	}
	good := []string{`x + 1`, `\frac{a}{b}`, `\text{当 }$x>0$`, `x = 1 \\`}
	for _, in := range good {
		r := Validate(in)
		for _, c := range r.Checks {
			if c.Type == "structure" && !c.OK {
				t.Fatalf("structure check should pass: %q (%s)", in, c.Detail)
			}
		}
	}
}

func TestRepairTrailingBreak(t *testing.T) {
	cases := map[string]string{
		`x = 1 \\`:      `x = 1 \\`,
		`\frac{a}{b}\\`: `\frac{a}{b}\\`,
		`x^2 + \`:       `x^2`,
		`x^2 +`:         `x^2`,
		`x = y +`:       `x = y`,
		`x > `:          `x`,
		`x = y + `:      `x = y`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairNestedDollars(t *testing.T) {
	cases := map[string]string{
		`$$x = $y$ + 1$$`:            `$$x = y + 1$$`,
		`$$ x = $y$ + $z$ $$`:        `$$ x = y + z $$`,
		`\text{当 $$x = $y$ + 1$$ 时}`: `\text{当 }$$x = y + 1$$\text{ 时}`,
		`\text{当 $x>0`:               `\text{当 }$x>0$`,
		`x^2 + $y$`:                  `x^2 + $y$`,
		`x = \$5 + 1`:                `x = \$5 + 1`,
		`$$x+1`:                      `$$x+1`,
		`x + $y`:                     `x + $y`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairEmptyStructures(t *testing.T) {
	cases := map[string]string{
		`\frac{}{x}`:  `x`,
		`\frac{x}{}`:  `x`,
		`x + \sqrt{}`: `x`,
		`x_{}`:        `x`,
		`x^{}`:        `x`,
		`\frac{}{}`:   ``,
		`\frac{a}{b}`: `\frac{a}{b}`,
		`\sqrt{x}`:    `\sqrt{x}`,
		`x^{2}`:       `x^{2}`,
		`x_{1}`:       `x_{1}`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairEnvMismatch(t *testing.T) {
	cases := map[string]string{
		`\begin{aligned} x \end{cases}`: `\begin{aligned} x \end{aligned}`,
		`\begin{cases} x \end{aligned}`: `\begin{cases} x \end{cases}`,
		`\begin{a} x \end{b}`:           `\begin{a} x \end{a}`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairExtraEnd(t *testing.T) {
	cases := map[string]string{
		`\begin{aligned} x \end{aligned} \end{cases}`: `\begin{aligned} x \end{aligned}`,
		`x \end{cases}`: `x`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairLeftRight(t *testing.T) {
	cases := map[string]string{
		`\left( x + 1`:             `\left( x + 1\right.`,
		`\left[ x \right] \left(`:  `\left[ x \right] \left(\right.`,
		`\left( x \right) \right]`: `\left( x \right)`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairTextWrappedMath(t *testing.T) {
	cases := map[string]string{
		`\text{2a = 3, 2c = 2}`:  `$2a = 3, 2c = 2$`,
		`2a = 3, \text{ 2c = 2}`: `2a = 3, $2c = 2$`,
		`\text{a = b, c = d}`:    `$a = b, c = d$`,
		`\text{ 2c = 2}`:         `$2c = 2$`,
		`\text{已知 a = 1, b = 2}`: `\text{已知 }$a = 1, b = 2$`,
		`\text{当 $x>0`:           `\text{当 }$x>0$`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}

func TestRepairDanglingOps(t *testing.T) {
	cases := map[string]string{
		`x^2 +`:   `x^2`,
		`x^`:      `x`,
		`x_`:      `x`,
		`x = y +`: `x = y`,
		`x > `:    `x`,
		`x^2`:     `x^2`,
		`x^{2}`:   `x^{2}`,
		`x_1`:     `x_1`,
	}
	for in, want := range cases {
		r := Repair(in)
		if r.Latex != want {
			t.Fatalf("repair(%q) = %q, want %q", in, r.Latex, want)
		}
	}
}
