import { Component } from '@angular/core';

@Component({
  selector: 'ss-app',
  template: `<div>
                <router-outlet></router-outlet>
              </div>`,
})
export class AppComponent {
  name = 'ShareSci';
}
