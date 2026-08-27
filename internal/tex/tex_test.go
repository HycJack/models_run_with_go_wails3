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
func TestValidateProse(t *testing.T) {
	r := Validate("hello this is some prose words")
	for _, c := range r.Checks {
		if c.Type == "content" && c.OK {
			t.Fatalf("prose should fail content check")
		}
	}
}
